#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"
#include <stdio.h>


//random init array 
#include <stdlib.h>
#include <time.h>

__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size) {

		c[gid] = a[gid] + b[gid];
	}

}

//verify output of gpu implementation 
void sum_array_cpu(int* a, int* b, int* c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}


void compare_arrays(int* a, int* b, int size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			printf("Arrays are different \n");
			return;
		}
	}
	printf("Arrays are same\n");
}

int main() {

	int size = 10000;
	int block_size = 128;

	int NO_BYTES = size * sizeof(int);

	//host and device pointers 
	int* h_a, * h_b, * gpu_results;
	
	int* h_c;

    // allocate memory
    h_a = (int*)malloc(NO_BYTES);
    h_b = (int*)malloc(NO_BYTES);
    gpu_results = (int*)malloc(NO_BYTES);
	h_c = (int*)malloc(NO_BYTES);



	if (h_a == NULL || h_b == NULL || gpu_results == NULL) {
		printf("Failed to allocate memory.\n");
		return 1;
	}



	//init host pointer
	time_t t;
	srand((unsigned)time(&t)); //create a seed using current time. 

	for (int i = 0; i < size; i++) {

		h_a[i] = (int)(rand() & 0xFF);
	}

	for (int i = 0; i < size; i++) {

		h_b[i] = (int)(rand() & 0xFF);
	}

	//perform addition in cpu
	sum_array_cpu(h_a, h_b, h_c, size);

	// set all elements of gpu_results to zero.
	memset(gpu_results, 0, NO_BYTES);

	//device pointers
	int* d_a, * d_b, * d_c;

	//allocate memory on GPU
	cudaMalloc((int**)&d_a, NO_BYTES);
	cudaMalloc((int**)&d_b, NO_BYTES);
	cudaMalloc((int**)&d_c, NO_BYTES);


	//transfer data to device. 
	
	// cudaMemcpy( DESTINATION, SOURCE, SIZE, DIRECTION OF TRANSFER);
	cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);


	//launch grid
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);


	//perform addition in gpu
	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	//transfer data back to the host, here gpu_results gets filled with the
	//results of the addition
	cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);

	//compare results of cpu and gpu implementation 
	compare_arrays(gpu_results, h_c, size);


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(gpu_results);
	free(h_a);
	free(h_b);


	cudaDeviceReset();
	return 0;
}