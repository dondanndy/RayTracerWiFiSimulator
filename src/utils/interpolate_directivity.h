#pragma once
#include "cuda_runtime.h"
#include "../cudaStructs/cuda3DMatrix.h"

__host__ __device__ float interpolate_directivity(float* data, cuda3DMatrix<float> data_prop, float azim, float elev);