#pragma once

#include "cuda_runtime.h"
#include <iostream>

template <typename T>
class cudaStack {
	//Device implementation of a stack.
private:
	T* data;
	size_t pos;
	size_t max_size;

public:
	__host__ __device__ cudaStack(size_t init_size)
		: data(nullptr), max_size(init_size), pos(0)
	{
		data = new T[init_size];
	};

	__host__ __device__ ~cudaStack() {
		if (data) delete[] data;
	};

	__host__ __device__ bool empty() const { return pos == 0; };
	__host__ __device__ size_t size() const { return pos; };

	__host__ __device__ bool add(const T& val) {
		if (pos < max_size) {
			data[pos] = val;
			pos++;
			return true;
		}
		else {
			return false;
		}
	};

	__host__ __device__ bool pop(T& val) {
		if (pos > 0) {
			val = data[pos-1];
			pos--;
			return true;
		}
		else {
			return false;
		}
	};
};