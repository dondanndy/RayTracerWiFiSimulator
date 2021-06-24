#pragma once

#include "cuda_runtime.h"

template<typename T1, typename T2>
class cudaPair {
/*
	Device implementation of a structure of two values.
*/

private:
	T1 first_val;
	T2 second_val;

public:
	__host__ __device__ cudaPair() {};

	__host__ __device__ cudaPair(T1 first, T2 second) 
		: first_val(first), second_val(second)
	{};

	__host__ __device__ cudaPair(T1&& first, T2 second)
		: first_val(std::move(first)), second_val(second)
	{};

	//YAGNI: No more constructors for the time being.

	__host__ __device__ T1 first() const { return first_val; };
	__host__ __device__ T2 second() const { return second_val; };
};