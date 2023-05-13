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

	int size = 100000000; 
	int block_size = 128;

	int NO_BYTES = size * sizeof(int);

	//runtime error handeling
	cudaError error;

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
	//timers
	clock_t start_cpu, end_cpu;

	//perform addition in cpu
	start_cpu = clock();
	sum_array_cpu(h_a, h_b, h_c, size);
	end_cpu = clock();


	// set all elements of gpu_results to zero.
	memset(gpu_results, 0, NO_BYTES);

	//device pointers
	int* d_a, * d_b, * d_c;

	//allocate memory on GPU
	error = cudaMalloc((int**)&d_a, NO_BYTES);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}
	error = cudaMalloc((int**)&d_b, NO_BYTES);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}
	error = cudaMalloc((int**)&d_c, NO_BYTES);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}


	//transfer data to device. 
	clock_t h2d_s, h2d_e;
	h2d_s = clock();
	// cudaMemcpy( DESTINATION, SOURCE, SIZE, DIRECTION OF TRANSFER);
	error = cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
	h2d_e = clock();

	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}
	error = cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}


	//launch grid
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);

	//timers
	clock_t start_gpu, end_gpu;
	start_gpu = clock();
	//perform addition in gpu
	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();
	end_gpu = clock();



	clock_t start_d2h, end_d2h;
	//transfer data back to the host, here gpu_results gets filled with the
	//results of the addition
	start_d2h = clock();
	error = cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}
	end_d2h = clock();
	//compare results of cpu and gpu implementation 
	compare_arrays(gpu_results, h_c, size);


	printf("CPU execution time: %1.50f \n",
		(double)((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC));

	printf("GPU execution time: %1.50f \n",
		(double)((double)(end_gpu - start_gpu) / CLOCKS_PER_SEC));

	printf("CPU to GPU transfer time: %1.50f \n",
		(double)((double)(h2d_e - h2d_s) / CLOCKS_PER_SEC));

	printf("GPU to CPU transfer time: %1.50f \n",
		(double)((double)(end_d2h - start_d2h) / CLOCKS_PER_SEC));





	error = cudaFree(d_a);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}
	error = cudaFree(d_b);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}
	error = cudaFree(d_c);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
	}

	free(gpu_results);
	free(h_a);
	free(h_b);


	cudaDeviceReset();
	return 0;
}
