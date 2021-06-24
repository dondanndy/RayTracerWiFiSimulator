#pragma once
#include "vec3.h"
#include "wall.h"
#include "ray.h"

class box {
public:
	Wall walls[6];
	point3 origin;
	float side;

public:
	box(){}
	__host__ __device__ box(const point3& og, const float side);

	__host__ __device__ bool hit(const Ray& r, float& final_power) const;
};
