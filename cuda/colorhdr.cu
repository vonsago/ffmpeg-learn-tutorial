#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>


__global__ void doGPU(uint8_t * data, uint8_t * dst_data, int length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= length)
        return;
    dst_data[i] = data[i];
}

int doitgpu(uint8_t * data, int width, int height, int format, uint8_t *ret, int length) {
    uint8_t * cuda_data;
    uint8_t * dst_data;

    cudaMalloc(&cuda_data, length);
    cudaMalloc(&dst_data, length);
    cudaMemcpy(cuda_data, data, length, cudaMemcpyHostToDevice);

    int blocks = (length + 128-1)/128;
    int threads = 128;
    doGPU <<<blocks, threads>>>(cuda_data, dst_data, length);

    cudaMemcpy(ret, dst_data, length, cudaMemcpyDeviceToHost);
    cudaDeviceReset();

    return 0;
}
