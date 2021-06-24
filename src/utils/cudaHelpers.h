#pragma once

#include "cudaStructs/cuda3DMatrix.h"
#include "cuda_runtime.h"
#include <iostream>

template<typename T>
cudaError_t alloc_in_gpu(T** gpu_ptr, cuda3DMatrix<T> prop) {
    return cudaMalloc((void**)gpu_ptr, prop.size());
}

template<typename T>
cudaError_t copy_to_gpu(T* src, T* gpu, cuda3DMatrix<T> prop) {
    return cudaMemcpy((void*)gpu, (void*)src, prop.size(), cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t copy_to_gpu(T* src, T* gpu, size_t data_size) {
    return cudaMemcpy((void*)gpu, (void*)src, data_size * (size_t)sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t copy_from_gpu(T* src, T* gpu, cuda3DMatrix<T> prop) {
    return cudaMemcpy((void*)src, (void*)gpu, prop.size(), cudaMemcpyDeviceToHost);
}

template<typename T>
T* alloc_and_copy_to_gpu(T* src, cuda3DMatrix<T> prop) {
    T* gpu;
    cudaError_t err;

    err = cudaMalloc((void**)&gpu, prop.size());

    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc has failed." << std::endl;
        return nullptr;
    }

    err = cudaMemcpy((void*)gpu, (void*)src, prop.size(), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy (cpu->gpu) has failed. Code = " << err << std::endl;
        return nullptr;
    }

    return gpu;
}


template<typename T>
bool alloc_and_copy_to_gpu(T* src, T** gpu_ptr, cuda3DMatrix<T> prop) {
    cudaError_t err;

    err = cudaMalloc((void**)gpu_ptr, prop.size());

    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc has failed. Code = " << err << std::endl;
        return false;
    }

    err = cudaMemcpy((void*)(*gpu_ptr), (void*)src, prop.size(), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy (cpu->gpu) has failed. Code = " << err << std::endl;
        return false;
    }

    return true;
}

template<typename T>
bool alloc_and_copy_to_gpu(T* src, T** gpu_ptr, size_t data_size) {
    cudaError_t err;

    err = cudaMalloc((void**)gpu_ptr, data_size * (size_t)sizeof(T));

    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc has failed. Code = " << err << std::endl;
        return false;
    }

    err = cudaMemcpy((void*)(*gpu_ptr), (void*)src, data_size * (size_t)sizeof(T), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy (cpu->gpu) has failed. Code = " << err << std::endl;
        return false;
    }

    return true;
}

