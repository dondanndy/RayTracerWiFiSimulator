#pragma once

#include "cuda_runtime_api.h"
#include <memory>
#include <vector>
#include <iostream>

template<typename T>
struct cudaVector {
/*
	Wrapper to hold cpu and gpu data.
*/

private:
	T* gpu_memory;
	std::vector<T> vec;

public:
	cudaVector() : gpu_memory(nullptr) {};
	
	cudaVector(const std::vector<T>& v)
		: vec(v), gpu_memory(nullptr)
	{}

	cudaVector(std::vector<T>&& v)
		: vec(std::move(v)), gpu_memory(nullptr)
	{}

	~cudaVector() {
		if (gpu_memory) {
			cudaFree(gpu_memory);
		}
	}

	T* gpu() { return gpu_memory; }; //Exposes the gpu memory.

	std::vector<T>& cpu(); //Exposes the cpu memory.

	void copy_to_gpu();
	void copy_from_gpu();
	size_t size();

private:
	cudaError_t init();
	cudaError_t copy();
	cudaError_t retrieve();
};

//Implementations

template <typename T>
cudaError_t cudaVector<T>::init() {
	return cudaMalloc((void**)&gpu_memory, vec.size() * (size_t)sizeof(T));
}

template <typename T>
cudaError_t cudaVector<T>::copy() {
	return cudaMemcpy((void*)gpu_memory, (void*)vec.data(), vec.size() * (size_t)sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
cudaError_t cudaVector<T>::retrieve() {
	return cudaMemcpy((void*)vec.data(), gpu_memory, vec.size() * (size_t)sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
std::vector<T>& cudaVector<T>::cpu(){
	return vec;
}

template <typename T>
size_t cudaVector<T>::size() {
	return vec.size();
}

template <typename T>
void cudaVector<T>::copy_to_gpu() {
	cudaError_t err;

	err = init();

	if (err != cudaSuccess) {
		std::cerr << "Error: cudaMalloc ha fallado." << std::endl;
	}

	err = copy();

	if (err != cudaSuccess) {
		std::cerr << "Error: cudaMemcpy (cpu->gpu) ha fallado." << std::endl;
	}
}

template <typename T>
void cudaVector<T>::copy_from_gpu() {
	cudaError_t err;

	err = retrieve();
	if (err != cudaSuccess) {
		std::cerr << "Error: cudaMemcpy (gpu->cpu) ha fallado." << std::endl;
	}
}