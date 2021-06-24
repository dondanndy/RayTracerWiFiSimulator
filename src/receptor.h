#pragma once

#include "cuda_runtime.h"
#include "parameters.h"
#include "vec3.h"
#include "ray.h"

#include "cudaStructs/cuda3DMatrix.h"
#include "utils/interpolate_directivity.h"


class Receptor {
	//Receptor as a Sphere
private:
	point3 origin;
	float radius;

	float azim_offset;
	float elev_offset;

public:
	Receptor() {};
	Receptor(point3& og, float r)
		: origin(og), radius(r), azim_offset(0), elev_offset(0)
	{}

	Receptor(point3& og, float r, float azimut, float elevation)
		: origin(og), radius(r), azim_offset(azimut), elev_offset(elevation)
	{}

	__host__ __device__ point3 get_origin() const { return origin; };

	__host__ __device__ bool hit(const Ray& r, float& distance, float& final_power, float* data, cuda3DMatrix<float> data_prop) const;
	__host__ __device__ bool intersect(const Ray& r, float& distance) const;

	__host__ __device__ vec3 get_normal(const point3& p) const;

private:
	__host__ __device__ float hit_azimut(const Ray& direction) const;
	__host__ __device__ float hit_elevation(const Ray& direction) const;
};