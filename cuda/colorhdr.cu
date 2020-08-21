#include <stdio.h>
#include <cuda_runtime.h>


__global__ void doGPU(float * data, float * dst_data){
    int i = threadIdx.x;
    printf("calc idx: i: %d\n", i);
    dst_data[i] = data[i];
}

int doitgpu(float * data, int width, int height, int format, float *ret) {
    float * cuda_data;
    float * dst_data = (float *)malloc(sizeof(data));

    cudaMalloc(&cuda_data, sizeof(data));
    cudaMemcpy(cuda_data, data, sizeof(data), cudaMemcpyHostToDevice);

    doGPU <<<1, sizeof(data)>>>(cuda_data, dst_data);

    cudaMemcpy(ret, dst_data, sizeof(data), cudaMemcpyDeviceToHost);
    cudaDeviceReset();

    return 0;
}