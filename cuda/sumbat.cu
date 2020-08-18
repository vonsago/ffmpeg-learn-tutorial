#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)
{
	const cudaError_t error = call;
	if (error != cudaSuccess)
	{
		printf("Error: %s:%d, ", __FILE__, __LINE__);
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
	}
}


__global__ void sumGPU(float *A, float *B, float *C){
	int i = threadIdx.x;
	printf("calc idx: i: %d\n", i);
	C[i] = A[i]+B[i];
}

int main(int argc, char **argv) {
	float A[3]={1,2,3};
	float B[3]={2,3,4};
	float C[3]={0,0,0};

	float* a_device, *b_device, *c_device;

	cudaMalloc((void**)&a_device, 3 * sizeof(float));
    	cudaMalloc((void**)&b_device, 3 * sizeof(float));
   	cudaMalloc((void**)&c_device, 3 * sizeof(float));

    	cudaMemcpy(a_device, A, 3 * sizeof(float), cudaMemcpyHostToDevice);
    	cudaMemcpy(b_device, B, 3 * sizeof(float), cudaMemcpyHostToDevice);

	sumGPU <<<1, 3>>>(a_device, b_device, c_device);

	cudaMemcpy(C, c_device, 3 * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0; i<3; i++)
		printf("%f\n",C[i]);
	cudaDeviceReset();
	return (0);
}

