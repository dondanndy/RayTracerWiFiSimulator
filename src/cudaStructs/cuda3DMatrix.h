#pragma once

#include "cuda_runtime.h"

template<typename T>
class cuda3DMatrix {
/*
	Structure to hold the data of a 3D Matrix.
	The intended use is to be able to allocate all the data at once in a malloc call and then, using the
	available methods in this class, access that data as if it had a 3D structure.
*/
private:
	size_t rows;
	size_t cols;
	size_t depth;

public:
	__host__ __device__ cuda3DMatrix(size_t rows, size_t cols, size_t depth)
		: rows(rows), cols(cols), depth(depth)
	{}

	__host__ __device__ T& at(T* data, size_t row, size_t col, size_t dep) {
		return data[col + cols * (row + dep * rows)];
	}

	__host__ __device__ size_t nrows() const { return rows; };
	__host__ __device__ size_t ncols() const { return cols; };
	__host__ __device__ size_t ndepth() const { return depth; };

	__host__ __device__ T& at(T* data, size_t row, size_t col) {
		//Assumes depth=1, a 2D array.
		return data[col + cols * row];
	}

	__host__ __device__ size_t elements() {
		return rows * cols * depth;
	}

	__host__ __device__ size_t size() {
		return rows * cols * depth * sizeof(T);
	}

	__host__ __device__ cuda3DMatrix<T> slice_2D() {
		/*
			Information for a 2D slice of the 3D matrix.
		*/
		return cuda3DMatrix<T>(rows, cols, 1);
	}
};
