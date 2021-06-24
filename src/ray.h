#pragma once
#include "cuda_runtime.h"
#include "vec3.h"
#include "parameters.h"

class Ray {
private:
    point3 orig;
    vec3 dir;
    float pow;

public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const point3& origin, const vec3& direction, float power)
        : orig(origin), dir(unit_vector(direction)), pow(power)
    {}

    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ float power() const { return pow; }
    __host__ __device__ void set_power(float power) { pow = power; }


    __host__ __device__ void set_direction(vec3 direction) { dir = direction; }
    __host__ __device__ vec3 direction() const { return dir; }

    __host__ __device__ point3 at(float t) const {
        return orig + t * dir;
    }

    __host__ __device__ float get_decay(float dist) const {
        //TODO: Más parámetros.
        return pow * (0.12 / (4.0 * PI * dist)) * (0.12 / (4.0 * PI * dist));
    }
};