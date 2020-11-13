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

__device__ float yuv2rgb_REC_2020_NCL[9] = {1.000000, 1.4196651e-17, 1.47459996, 1.000000, -0.164553121,  -0.571353137, 1.000000, 1.88139999, 5.67866042e-17};
__device__ float rgb2yuv_REC_709[9] = {0.212599993, 0.715200007, 0.0722000003, -0.114572108, -0.385427892, 0.500000, 0.500000, -0.454152912, -0.0458470918};
__device__ float gamutMa[9] = {1.66049099, -0.58764112, -0.0728498623, -0.124550477, 1.13289988, -0.00834942237, -0.0181507636, -0.100578897, 1.11872971};

struct FilterContext {
    unsigned filter_width;
    unsigned stride;
    float *data;
    int *left;
};

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

__device__ uint16_t clamp(uint16_t v, uint16_t min = 0, uint16_t max = 255) {
    if (v > max)
        return max;
    if (v < min)
        return min;
    return v;
}

__device__ float rec_709_oetf(float x){
    //[3] ToGammaLutOperationAVX2
    //if (x < 4.5f * REC709_BETA)
    //    x = x / 4.5f;
    //else
    //    x = pow((x + (REC709_ALPHA - 1.0f)) / REC709_ALPHA, 1.0f / 0.45f);
    //return x;
    //rec_1886_inverse_eotf
    return x < 0.0f ? 0.0f : pow(x, 1.0f / 2.4f);

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


__device__ void linear_to_gamma(float r, float g, float b, float *rr, float *rg, float *rb) {
    *rr = rec_709_oetf(r);
    *rg = rec_709_oetf(g);
    *rb = rec_709_oetf(b);
}

__device__ void gamma_to_linear(float r, float g, float b, float *rr, float *rg, float *rb) {
    *rr = 100.0*st_2084_eotf(r);
    *rg = 100.0*st_2084_eotf(g);
    *rb = 100.0*st_2084_eotf(b);
}

__device__ float int16_2floaty(uint16_t x){
    return float(x) * 0.00114155246 + -0.0730593577;
}
__device__ float int16_2float(uint16_t x){
    return float(x) * 0.00111607148 + -0.571428597;
}

__device__ uint16_t float_2int16y(float x){
    x = x * 876 + 64;
    float d = 0;
    x+=d;

    if(x<0.0f)
        x = 0.0f;
    if(x<float(1UL<<10)-1)
        return x;
    return float(1UL<<10)-1;
}

__device__ uint16_t float_2int16(float x){
    x = x * 876 + 512;
    float d = 0;
    x+=d;

    if(x<0.0f)
        x = 0.0f;
    if(x<float(1UL<<10)-1)
        return x;
    return float(1UL<<10)-1;
}

__device__ void yuv_to_rgb(uint16_t *y, uint16_t *u, uint16_t *v, uint16_t *ry, uint16_t *ru, uint16_t *rv) {
    // int to float

    float iy,iu,iv;
    iy = int16_2floaty(*y);
    iu = int16_2float(*u);
    iv = int16_2float(*v);

    float r = iy * yuv2rgb_REC_2020_NCL[0] + iu * yuv2rgb_REC_2020_NCL[1] + iv * yuv2rgb_REC_2020_NCL[2];
    float g = iy * yuv2rgb_REC_2020_NCL[3] + iu * yuv2rgb_REC_2020_NCL[4] + iv * yuv2rgb_REC_2020_NCL[5];
    float b = iy * yuv2rgb_REC_2020_NCL[6] + iu * yuv2rgb_REC_2020_NCL[7] + iv * yuv2rgb_REC_2020_NCL[8];

    float lr, lg, lb;
    gamma_to_linear(r,g,b, &lr, &lg, &lb);

    r = lr * gamutMa[0] + lg * gamutMa[1] + lb * gamutMa[2];
    g = lr * gamutMa[3] + lg * gamutMa[4] + lb * gamutMa[5];
    b = lr * gamutMa[6] + lg * gamutMa[7] + lb * gamutMa[8];

    float gr, gg, gb;
    linear_to_gamma(r,g,b, &gr, &gg, &gb);
    //gr = r; gg=g; gb=b;

    iy = gr * rgb2yuv_REC_709[0] + gg * rgb2yuv_REC_709[1] + gb * rgb2yuv_REC_709[2];
    iu = gr * rgb2yuv_REC_709[3] + gg * rgb2yuv_REC_709[4] + gb * rgb2yuv_REC_709[5];
    iv = gr * rgb2yuv_REC_709[6] + gg * rgb2yuv_REC_709[7] + gb * rgb2yuv_REC_709[8];

    // float to int
    *ry = clamp(float_2int16y(iy), 16*4, 235*4);
    *ru = clamp(float_2int16(iu), 16*4, 240*4);
    *rv = clamp(float_2int16(iv), 16*4, 240*4);

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

__global__ void colorspace(float *src0, float *src1, float *src2, float *dst0, float *dst1, float *dst2, int w, int h){
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    if(i>w*h-1)
        return;
    float iy = src0[i];
    float iu = src1[i];
    float iv = src2[i];
    float r = iy * yuv2rgb_REC_2020_NCL[0] + iu * yuv2rgb_REC_2020_NCL[1] + iv * yuv2rgb_REC_2020_NCL[2];
    float g = iy * yuv2rgb_REC_2020_NCL[3] + iu * yuv2rgb_REC_2020_NCL[4] + iv * yuv2rgb_REC_2020_NCL[5];
    float b = iy * yuv2rgb_REC_2020_NCL[6] + iu * yuv2rgb_REC_2020_NCL[7] + iv * yuv2rgb_REC_2020_NCL[8];

    float lr, lg, lb;
    gamma_to_linear(r,g,b, &lr, &lg, &lb);

    r = lr * gamutMa[0] + lg * gamutMa[1] + lb * gamutMa[2];
    g = lr * gamutMa[3] + lg * gamutMa[4] + lb * gamutMa[5];
    b = lr * gamutMa[6] + lg * gamutMa[7] + lb * gamutMa[8];

    float gr, gg, gb;
    linear_to_gamma(r,g,b, &gr, &gg, &gb);
    //gr = r; gg=g; gb=b;

    dst0[i] = gr * rgb2yuv_REC_709[0] + gg * rgb2yuv_REC_709[1] + gb * rgb2yuv_REC_709[2];
    dst1[i] = gr * rgb2yuv_REC_709[3] + gg * rgb2yuv_REC_709[4] + gb * rgb2yuv_REC_709[5];
    dst2[i] = gr * rgb2yuv_REC_709[6] + gg * rgb2yuv_REC_709[7] + gb * rgb2yuv_REC_709[8];

    //yuv_to_rgb(&src0[i], &src1[i], &src2[i], &dst0[i], &dst1[i], &dst2[i]);
}

__global__ void resize_line_32_y(uint8_t *src, float *dst, int width, int height) {
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    if(i>width*height-1)
        return;
    dst[i] = int16_2floaty(src[i]);
}

__global__ void resize_line_32_uv(uint8_t *src, float *dst, int wi, int hi){
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    if(i>wi*hi-1)
        return;

    int h = int(i / wi);
    int v = int(i % wi);
    int w = int(wi / 2);
    if (h == 0 || h == hi-1){
        if (v % 2 == 0 || v == wi - 1)
            dst[i] = int16_2float(src[int(h/2)*w +int(v / 2)]);
        else
            dst[i] = int16_2float(src[int(h/2) * w + int(v / 2)]) * 0.5 + int16_2float(src[int(h / 2) * w + int(v / 2) + 1]) * 0.5;
    }
    else if (h % 2 == 0){
        if (v % 2 == 0 || v == wi - 1)
            dst[i] = int16_2float(src[int(h / 2 - 1) * w + int(v / 2)]) * 0.25 + int16_2float(src[int(h / 2) * w + int(v / 2)]) * 0.75;
        else{
            float x1 = int16_2float(src[int(h / 2 - 1) * w + int(v / 2)]) * 0.5 + int16_2float(src[int(h / 2 - 1) * w + int(v / 2) + 1]) * 0.5;
            float x2 = int16_2float(src[int(h / 2) * w + int(v / 2)]) * 0.5 + int16_2float(src[int(h / 2) * w + int(v / 2) + 1]) * 0.5;
            dst[i] = x1 * 0.25 + x2 * 0.75;
        }
    }
    else{
        if (v % 2 == 0 || v == wi - 1)
            dst[i] = int16_2float(src[int(h / 2) * w + int(v / 2)]) * 0.75 + int16_2float(src[int(h / 2 + 1) * w + int(v / 2)]) * 0.25;
        else{
            float x1 = int16_2float(src[int(h / 2) * w + int(v / 2)]) * 0.5 + int16_2float(src[int(h / 2) * w + int(v / 2) + 1]) * 0.5;
            float x2 = int16_2float(src[int(h / 2 + 1) * w + int(v / 2)]) * 0.5 + int16_2float(src[int(h / 2 + 1) * w + int(v / 2) + 1]) * 0.5;
            dst[i] = x1 * 0.75 + x2 * 0.25;
        }
    }
}

__global__ void stamp_line_32_y(float *src, uint8_t *dst, int wi, int hi){
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    if(i>wi*hi-1)
        return;
    dst[i] = clamp(float_2int16y(src[i]), 16*4, 235*4);
}

__global__ void stamp_line_32_uv(float *src, uint8_t *dst, int wi, int hi){
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    if(i>wi*hi-1)
        return;
    float x1,x2,x3;
    x1=x2=x3=0;
    int h = int(i / wi);
    int v = int(i % wi);
    int w = int(wi * 2);

    int sh = (h+1)/2;
    int sv = (v+1)/2;

    if (h == hi -1){
        float x1 = src[sh * w + sv] * 0.5 + src[(sh + 1) * w + sv] * 0.375 + src[(sh + 2) * w + sv] * 0.125;
        float x2 = src[sh * w + sv + 1] * 0.5 + src[(sh + 1) * w + sv + 1] * 0.375 + src[(sh + 2) * w + sv + 1] * 0.125;
        float x3 = src[sh * w + sv + 2] * 0.5 + src[(sh + 1) * w + sv + 2] * 0.375 + src[(sh + 2) * w + sv + 2] * 0.125;
    }
    else{
        float x1 = src[sh * w + sv] * 0.125 + src[(sh + 1) * w + sv] * 0.375 + src[(sh + 2) * w + sv] * 0.375 + src[(sh + 3) * w + sv] * 0.125;
        float x2 = src[sh * w + sv + 1] * 0.125 + src[(sh + 1) * w + sv + 1] * 0.375 + src[(sh + 2) * w + sv + 1] * 0.375 + src[(sh + 3) * w + sv + 1] * 0.125;
        float x3 = src[sh * w + sv + 2] * 0.125 + src[(sh + 1) * w + sv + 2] * 0.375 + src[(sh + 2) * w + sv + 2] * 0.375 + src[(sh + 3) * w + sv + 2] * 0.125;
    }
    dst[i] = clamp(float_2int16(x1 * 0.25 + x2 * 0.5 + x3 * 0.25), 16*4, 240*4);
}

extern "C" {
int
doitgpu(uint8_t *data0, uint8_t *data1, uint8_t *data2, int *linesize, int width, int height, int format, uint8_t *out0,
        uint8_t *out1, uint8_t *out2) {
    fprintf(stderr, "=====gpu====\n");
    uint8_t *cuda_data0, *cuda_data1, *cuda_data2;
    uint8_t *dst_data0, *dst_data1, *dst_data2;
    float *dst_w0, *dst_w1, *dst_w2, *dst_prt0, *dst_prt1, *dst_prt2;
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

    cudaMalloc(&dst_w0, 4 * width * height);
    cudaMalloc(&dst_w1, 4 * width * height);
    cudaMalloc(&dst_w2, 4 * width * height);

    cudaMalloc(&dst_prt0, 4 * width * height);
    cudaMalloc(&dst_prt1, 4 * width * height);
    cudaMalloc(&dst_prt2, 4 * width * height);

    int blocks = (width * height +128 - 1)/128;
    int threads = 128;
    resize_line_32_y<<<blocks, threads>>>(cuda_data0, dst_w0, width, height);
    resize_line_32_uv<<<blocks, threads>>>(cuda_data1, dst_w1, width, height);
    resize_line_32_uv<<<blocks, threads>>>(cuda_data2, dst_w2, width, height);
    fprintf(stderr, "--resized\n");

    colorspace<<<blocks, threads>>>(dst_w0, dst_w1, dst_w2, dst_prt0, dst_prt1, dst_prt2, width, height);
    fprintf(stderr, "--colorspace\n");

    stamp_line_32_y<<<blocks, threads>>>(dst_prt0, dst_data0, width, height);

    blocks = (width * height / 4 + 128 - 1) / 128;
    threads = 128;
    //calc_colorspace <<<blocks, threads>>>(cuda_data0, cuda_data1, cuda_data2, width, height, stride_y, stride_uv,
    //                                      dst_data0, dst_data1, dst_data2, 0);
    stamp_line_32_uv<<<blocks, threads>>>(dst_prt1, dst_data1, width/2, height/2);
    stamp_line_32_uv<<<blocks, threads>>>(dst_prt2, dst_data2, width/2, height/2);
    fprintf(stderr, "--over\n");


    cudaMemcpy(out0, dst_data0, length0, cudaMemcpyDeviceToHost);
    cudaMemcpy(out1, dst_data1, length1, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, dst_data2, length2, cudaMemcpyDeviceToHost);
    cudaDeviceReset();

    return 0;
}

}