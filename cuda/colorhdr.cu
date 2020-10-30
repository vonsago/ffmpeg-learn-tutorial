#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

#define FLT_MIN 1.175494351e-38F

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

__device__ float REC709_ALPHA = 1.09929682680944f;
__device__ float REC709_BETA = 0.018053968510807f;

__device__ float ST2084_M1 = 0.1593017578125f;
__device__ float ST2084_M2 = 78.84375f;
__device__ float ST2084_C1 = 0.8359375f;
__device__ float ST2084_C2 = 18.8515625f;
__device__ float ST2084_C3 = 18.6875f;

__device__ float yuv2rgb_REC_2020_NCL[9] = {1.000000, 0.000000, 1.474600, 1.000000, -0.164553, -0.571353, 1.000000, 1.881400, 0.000000};
__device__ float rgb2yuv_REC_709[9] = {0.212600, 0.715200, 0.072200, -0.114572, -0.385428, 0.500000, 0.500000, -0.454153, -0.045847};
__device__ float gamutMa[9] = {1.660491, -0.587641, -0.072850, -0.124550, 1.132900, -0.008349, -0.018151, -0.100579, 1.118730};

__global__ void doGPU(uint8_t *data, uint8_t *dst_data, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= length)
        return;
    dst_data[i] = data[i];
}

__device__ void save_yuv(uint8_t *data, uint8_t *dst_data, int i) {
    for (int x = 0; x < 4; x++) {
        dst_data[x] = data[x];
    }
}

__device__ float clamp(float v, float min = 0, float max = 255) {
    if (v > max)
        return max;
    if (v < min)
        return min;
    return v;
}

__device__ float rec_709_oetf(float x){
    //[3] ToGammaLutOperationAVX2
    if (x < 4.5f * REC709_BETA)
        x = x / 4.5f;
    else
        x = pow((x + (REC709_ALPHA - 1.0f)) / REC709_ALPHA, 1.0f / 0.45f);

    return x;
}

__device__ float st_2084_eotf(float x){
    // [1] ToLinearLutOperationAVX2
    // Filter negative values to avoid NAN.
    if (x > 0.0f) {
        float xpow = pow(x, 1.0f / ST2084_M2);
        float num = max(xpow - ST2084_C1, 0.0f);
        float den = max(ST2084_C2 - ST2084_C3 * xpow, FLT_MIN);
        x = pow(num / den, 1.0f / ST2084_M1);
    } else {
        x = 0.0f;
    }
    return x;
}


__device__ void gamma_to_linear(float r, float g, float b, float *rr, float *rg, float *rb) {
    *rr = rec_709_oetf(r);
    *rg = rec_709_oetf(g);
    *rb = rec_709_oetf(b);
}

__device__ void linear_to_gamma(float r, float g, float b, float *rr, float *rg, float *rb) {
    *rr = st_2084_eotf(r);
    *rg = st_2084_eotf(g);
    *rb = st_2084_eotf(b);
}

__device__ void yuv_to_rgb(uint16_t *y, uint16_t *u, uint16_t *v, uint16_t *ry, uint16_t *ru, uint16_t *rv) {
    float r = *y * yuv2rgb_REC_2020_NCL[0] + *u * yuv2rgb_REC_2020_NCL[1] + *v * yuv2rgb_REC_2020_NCL[2];
    float g = *y * yuv2rgb_REC_2020_NCL[3] + *u * yuv2rgb_REC_2020_NCL[4] + *v * yuv2rgb_REC_2020_NCL[5];
    float b = *y * yuv2rgb_REC_2020_NCL[6] + *u * yuv2rgb_REC_2020_NCL[7] + *v * yuv2rgb_REC_2020_NCL[8];

    float lr, lg, lb;
    gamma_to_linear(r,g,b, &lr, &lg, &lb);

    r = lr * gamutMa[0] + lg * gamutMa[1] + lb * gamutMa[2];
    g = lr * gamutMa[3] + lg * gamutMa[4] + lb * gamutMa[5];
    b = lr * gamutMa[6] + lg * gamutMa[7] + lb * gamutMa[8];

    float gr, gg, gb;
    linear_to_gamma(r,g,b, &gr, &gg, &gb);

    *ry = gr * rgb2yuv_REC_709[0] + gg * rgb2yuv_REC_709[1] + gb * rgb2yuv_REC_709[2];
    *ru = gr * rgb2yuv_REC_709[3] + gg * rgb2yuv_REC_709[4] + gb * rgb2yuv_REC_709[5];
    *rv = gr * rgb2yuv_REC_709[6] + gg * rgb2yuv_REC_709[7] + gb * rgb2yuv_REC_709[8];

    //*ry = *y;
    //*ru = *u;
    //*rv = *v;
}


__global__ void
calc_colorspace(uint8_t *data0, uint8_t *data1, uint8_t *data2, int w, int h, int stride_y, int stride_uv,
                uint8_t *dst_data0, uint8_t *dst_data1, uint8_t *dst_data2, int colorspace) {
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    int iy = i/(w/2)*2;
    int ix = (i % (w/2)) *2;

    if (ix >= w || iy >= h)
        return;

    int ofy = iy * stride_y + ix;
    int ofuv = (iy>>1) * stride_uv + ix/2;

    uint16_t * srcy = (uint16_t *)data0;
    uint16_t * srcu = (uint16_t *)data1;
    uint16_t * srcv = (uint16_t *)data2;

    uint16_t * dsty = (uint16_t *)dst_data0;
    uint16_t * dstu = (uint16_t *)dst_data1;
    uint16_t * dstv = (uint16_t *)dst_data2;

    yuv_to_rgb(&srcy[ofy], &srcu[ofuv], &srcv[ofuv], &dsty[ofy], &dstu[ofuv], &dstv[ofuv]);
    yuv_to_rgb(&srcy[ofy + 1], &srcu[ofuv], &srcv[ofuv], &dsty[ofy + 1], &dstu[ofuv], &dstv[ofuv]);
    yuv_to_rgb(&srcy[ofy + stride_y], &srcu[ofuv], &srcv[ofuv], &dsty[ofy + stride_y], &dstu[ofuv], &dstv[ofuv]);
    yuv_to_rgb(&srcy[ofy + stride_y +1], &srcu[ofuv], &srcv[ofuv], &dsty[ofy + stride_y + 1], &dstu[ofuv], &dstv[ofuv]);
}

extern "C" {
int
doitgpu(uint8_t *data0, uint8_t *data1, uint8_t *data2, int *linesize, int width, int height, int format, uint8_t *out0,
        uint8_t *out1, uint8_t *out2) {
    fprintf(stderr, "=-=-=-=-=-=-=-=-=--=-=-=-=-==-=-=-==-=-=-=-=-=-=-=-=-=\n");
    uint8_t *cuda_data0, *cuda_data1, *cuda_data2;
    uint8_t *dst_data0, *dst_data1, *dst_data2;
    int length0 = linesize[0] * height;
    int length1 = linesize[1] * height >> 1;
    int length2 = linesize[2] * height >> 1;
    int stride_y = linesize[0]>>1;
    int stride_uv = linesize[1]>>1;

    cudaMalloc(&cuda_data0, length0);
    cudaMalloc(&cuda_data1, length1);
    cudaMalloc(&cuda_data2, length2);
    cudaMalloc(&dst_data0, length0);
    cudaMalloc(&dst_data1, length1);
    cudaMalloc(&dst_data2, length2);
    cudaMemcpy(cuda_data0, data0, length0, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_data1, data1, length1, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_data2, data2, length2, cudaMemcpyHostToDevice);

    int blocks = (width * height / 4 + 128 - 1) / 128;
    int threads = 128;
    //doGPU <<<blocks, threads>>>(cuda_data0, dst_data0, length0);
    //doGPU <<<blocks, threads>>>(cuda_data1, dst_data1, length1);
    //doGPU <<<blocks, threads>>>(cuda_data2, dst_data2, length2);
    calc_colorspace <<<blocks, threads>>>(cuda_data0, cuda_data1, cuda_data2, width, height, stride_y, stride_uv,
                                          dst_data0, dst_data1, dst_data2, 0);

    cudaMemcpy(out0, dst_data0, length0, cudaMemcpyDeviceToHost);
    cudaMemcpy(out1, dst_data1, length1, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, dst_data2, length2, cudaMemcpyDeviceToHost);
    cudaDeviceReset();

    return 0;
}

}