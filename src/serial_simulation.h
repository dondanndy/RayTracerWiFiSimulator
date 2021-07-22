#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "parameters.h"
#include "vec3.h"
#include "ray.h"
#include "wall.h"
#include "Receptor.h"

#include "cudaStructs/cudaStack.h"
#include "cudaStructs/cudaPair.h"
#include "cudaStructs/cuda3DMatrix.h"

/*
	Functions to run the simulation as usual in the cpu.

	That means a double loop to cover the full range of angles to run on the cpu.
*/


void run_serial_simulation(const std::vector<Wall>& map, const std::vector<Receptor>& receptors,
	                       float* antenna_pattern, cuda3DMatrix<float> antenna_pattern_props,
	                       float* results);

void run_ray(const Ray& r, const std::vector<Wall>& map, const std::vector<Receptor>& tx,
	         float* data, cuda3DMatrix<float> data_prop,
	         float* directivity, cuda3DMatrix<float> directivity_prop);
