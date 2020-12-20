#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

using namespace std;

#define N 1024
#define GRID_SIZE 128
#define BLOCK_SIZE 128

__global__ void PiCalcGPU(float* res, curandState* states) {
	unsigned long index = threadIdx.x + blockDim.x * blockIdx.x;
	int V = 0;
	float x, y;
	//curand для генерации случайных чисел на GPU
	curand_init(index, index, 0, &states[index]);

	for (int i = 0; i < N; i++) {
		//создаем последовательности значений x и y
		x = curand_uniform(&states[index]);
		y = curand_uniform(&states[index]);
		//рассчитываем V для значений
		V += (x * x + y * y <= 1.0f);
	}
	res[index] = 4.0f * V / (float)N;
}

float PiCalcGPU(long n) {
	float x, y;
	long V = 0;
	for (long i = 0; i < n; i++) {
		x = rand() / (float)RAND_MAX;
		y = rand() / (float)RAND_MAX;
		V += (x * x + y * y <= 1.0f);
	}
	return 4.0f * V / n;
}

int main(int argc, char* argv[]) {
	setlocale(LC_ALL, "Russian");
	//переменные времени
	clock_t start, stop;
	float host[GRID_SIZE * BLOCK_SIZE];
	float* device;
	curandState* curand;

	//Вычисление на GPU
	//Старт
	start = clock();
	//Выделение памяти
	cudaError_t cuerr = cudaMalloc((void**)&device, GRID_SIZE * BLOCK_SIZE * sizeof(float));
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate device: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}
	cuerr = cudaMalloc((void**)&curand, BLOCK_SIZE * GRID_SIZE * sizeof(curandState));
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate device: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}
	//Запуск ядра
	PiCalcGPU <<< GRID_SIZE, BLOCK_SIZE >>> (device, curand);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}
	//Копируем результат с девайса на хост
	cuerr = cudaMemcpy(host, device, GRID_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy from device to host: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}
	float PI_GPU = 0;
	for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
		PI_GPU += host[i];
	}
	PI_GPU /= (GRID_SIZE * BLOCK_SIZE);
	stop = clock();

	printf("GPU PI = %f\n", PI_GPU);
	printf("Время работы на GPU %f c\n", (stop - start) / (float)CLOCKS_PER_SEC);

	//Вычисление на CPU
	start = clock();
	float cpuPI = PiCalcGPU(GRID_SIZE * BLOCK_SIZE * N);
	stop = clock();
	printf("CPU PI = %f\n", cpuPI);
	printf("Время работы на СPU %f c.\n", (stop - start) / (float)CLOCKS_PER_SEC);

	return 0;
}
