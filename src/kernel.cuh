#pragma once

#include <vector>
//#include <string>

#include "cuda_runtime.h"
#include "parameters.h"
#include "vec3.h"
#include "ray.h"
#include "wall.h"
#include "Receptor.h"

#include "utils/cudaHelpers.h"
#include "utils/interpolate_directivity.h"

#include "cudaStructs/cudaStack.h"
#include "cudaStructs/cudaPair.h"
#include "cudaStructs/cuda3DMatrix.h"

/*
	This file contains the functions declarations for the execution of the code in the GPU.

	The first function will setup the memory and copy the data into the GPU, the next three are the actual compuatation kernel.
*/

void setup_and_run_CUDA_simulation(std::vector<Wall> map, std::vector<Receptor> receptors, float* antenna_pattern, cuda3DMatrix<float> antenna_pattern_props, float* results);

//Main kernel
__global__ void run_simulation(point3 origin, Wall* map, size_t map_size,
	                           Receptor* tx, size_t tx_size,
	                           float* tx_power, cuda3DMatrix<float> tx_props,
	                           float* directivity, cuda3DMatrix<float> directivity_props,
	                           float ang_step);

__device__ void run_ray(Ray r, Wall* map, size_t map_size, Receptor* tx, size_t tx_size, float* tx_power, cuda3DMatrix<float> tx_power_prop, float* directivity, cuda3DMatrix<float> directivity_props);

__device__ void accumulate_block_data(float* full_data, cuda3DMatrix<float> full_data_prop, float* block_data, cuda3DMatrix<float> block_data_prop);

__device__ void fill_3D_matrix(float val, float* data, cuda3DMatrix<float> data_props);