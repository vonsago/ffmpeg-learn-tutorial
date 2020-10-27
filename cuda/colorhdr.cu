#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

__device__ float FCC_KR = 0.3;
__device__ float FCC_KB = 0.11;
__device__ float SMPTE_240M_KR = 0.212;
__device__ float SMPTE_240M_KB = 0.087;
__device__ float REC_601_KR = 0.299;
__device__ float REC_601_KB = 0.114;
__device__ float REC_709_KR = 0.2126;
__device__ float REC_709_KB = 0.0722;
__device__ float REC_2020_KR = 0.2627;
__device__ float REC_2020_KB = 0.0593;

__device__ struct Matrix_bt709 {
    float a = 0.2126;
    float b = 0.7152;
    float c = 0.0722;
    float d = 1.8556;
    float e = 1.5748;
};

__device__ struct Matrix_bt2020 {
    float a = 0.2627;
    float b = 0.6780;
    float c = 0.0593;
    float d = 1.8814;
    float e = 1.4747;
};

__device__ float yuv2rgb_REC_2020_NCL[9] = {1.000000,0.000000,1.474600,1.000000,-0.164553,-0.571353,1.000000,1.881400,0.000000};


__global__ void doGPU(uint8_t * data, uint8_t * dst_data, int length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= length)
        return;
    dst_data[i] = data[i];
}

__device__ void save_yuv(uint8_t* data, uint8_t* dst_data, int i) {
    for(int x=0;x<4;x++){
        dst_data[x] = data[x];
    }
}

__device__ float clamp(float v, float min=0, float max=255) {
    if(v > max)
        return max;
    if(v < min)
        return min;
    return v;
}

__device__ void yuv_to_rgb(uint8_t *y, uint8_t *u, uint8_t *v, uint8_t *ry, uint8_t *ru, uint8_t *rv) {
    Matrix_bt2020 ma;
    Matrix_bt709 ma7;
    float r = clamp((*y-16) + ma.e * (*v-128));
    float g = clamp((*y-16) - (ma.a * ma.e / ma.b) * (*v-128) - (ma.c * ma.d /ma.b) * (*u-128));
    float b = clamp((*y-16) + ma.d * (*u-128));

    *ry = ma7.a * r + ma7.b * g + ma7.c * b;
    *ru = (b - *ry) * ma7.d + 128;
    *rv = (r - *ry) * ma7.e + 128;

    //*ry = *y;
    //*ru = *u;
    //*rv = *v;
}


__global__ void calc_colorspace(uint8_t *data0, uint8_t *data1, uint8_t *data2, int w, int h, int stride_y, int stride_uv, uint8_t *dst_data0, uint8_t *dst_data1, uint8_t *dst_data2, int colorspace) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int iy = i / w;
    int ix = i % w;
    if (ix >= w || iy >=h)
        return;

    int ofy = int(iy*4) * stride_y + ix;
    int ofuv = int(iy*2) * stride_uv + ix;

    //dst_data0[ofy] = data0[ofy];
    //dst_data0[ofy+stride_y] = data0[ofy+stride_y];
    //dst_data0[ofy+stride_y*2] = data0[ofy+stride_y*2];
    //dst_data0[ofy+stride_y*3] = data0[ofy+stride_y*3];
    //
    //dst_data1[ofuv] = data1[ofuv];
    //dst_data2[ofuv] = data2[ofuv];

    //dst_data1[ofuv+stride_uv] = data1[ofuv+stride_uv];
    //dst_data2[ofuv+stride_uv] = data2[ofuv+stride_uv];
    yuv_to_rgb(&data0[ofy], &data1[ofuv], &data2[ofuv], &dst_data0[ofy], &dst_data1[ofuv], &dst_data2[ofuv]);
    yuv_to_rgb(&data0[ofy+stride_y], &data1[ofuv], &data2[ofuv], &dst_data0[ofy+stride_y], &dst_data1[ofuv], &dst_data2[ofuv]);
    yuv_to_rgb(&data0[ofy+stride_y*2], &data1[ofuv], &data2[ofuv], &dst_data0[ofy+stride_y*2], &dst_data1[ofuv], &dst_data2[ofuv]);
    yuv_to_rgb(&data0[ofy+stride_y*3], &data1[ofuv], &data2[ofuv], &dst_data0[ofy+stride_y*3], &dst_data1[ofuv], &dst_data2[ofuv]);

}

extern "C" {
int doitgpu(uint8_t * data0, uint8_t *data1, uint8_t *data2, int *linesize, int width, int height, int format,  uint8_t * out0, uint8_t *out1, uint8_t *out2) {
    fprintf(stderr, "=-=-=-=-=-=-=-=-=--=-=-=-=-==-=-=-==-=-=-=-=-=-=-=-=-=\n");
    uint8_t *cuda_data0, *cuda_data1, *cuda_data2;
    uint8_t *dst_data0, *dst_data1, *dst_data2;
    int length0 = linesize[0] * height;
    int length1 = linesize[1] * height >> 1;
    int length2 = linesize[2] * height >> 1;
    int stride_y = linesize[0];
    int stride_uv = linesize[1];

    cudaMalloc(&cuda_data0, length0);
    cudaMalloc(&cuda_data1, length1);
    cudaMalloc(&cuda_data2, length2);
    cudaMalloc(&dst_data0, length0);
    cudaMalloc(&dst_data1, length1);
    cudaMalloc(&dst_data2, length2);
    cudaMemcpy(cuda_data0, data0, length0, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_data1, data1, length1, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_data2, data2, length2, cudaMemcpyHostToDevice);

    int blocks = (width * height/4 + 128-1)/128;
    int threads = 128;
    //doGPU <<<blocks, threads>>>(cuda_data0, dst_data0, length0);
    //doGPU <<<blocks, threads>>>(cuda_data1, dst_data1, length1);
    //doGPU <<<blocks, threads>>>(cuda_data2, dst_data2, length2);
    calc_colorspace <<<blocks, threads>>>(cuda_data0, cuda_data1, cuda_data2, width, height, stride_y, stride_uv, dst_data0, dst_data1, dst_data2, 0);

    cudaMemcpy(out0, dst_data0, length0, cudaMemcpyDeviceToHost);
    cudaMemcpy(out1, dst_data1, length1, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, dst_data2, length2, cudaMemcpyDeviceToHost);
    cudaDeviceReset();

    return 0;
}

}