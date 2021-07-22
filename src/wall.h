#pragma once
#include "vec3.h"
#include "ray.h"
#include <cmath>

class Wall {
    /*
        Class to modelize a wall. It will just be a plane, so we would just need a normal.

        Each wall is defined by its 4 corners and the dielectrical permitivity for its interactions with the rays.
    */

public:
    point3 limits[4];
    vec3 normal;
    float permitivity;

public:
    __host__ __device__ Wall() {}

    __host__ __device__ Wall(point3 comps[4], float e)
        : limits{ comps[0], comps[1], comps[2], comps[3] }, permitivity(e)
    {
        calc_normal();
    }

    __host__ __device__ Wall(point3 x0, point3 x1, point3 y0, point3 y1, float e)
        : limits{ x0, x1, y0, y1 }, permitivity(e)
    {
        calc_normal();
    }

    __host__ __device__ vec3 get_normal() const { return normal; }

    //bool hit(const ray& r, double t_min, double t_max, point3& rec) const;

    __host__ __device__ bool hit(const Ray& r, float& distance, float& final_power) const;

    __host__ __device__ void get_hit_rays(const vec3& in, const point3& hit_point, const float power, Ray& reflected, Ray& transmitted) const;

    __host__ __device__ float intersect(const Ray& r) const;
    __host__ __device__ bool check_lims(const point3& p) const;

private:
    __host__ __device__ void calc_normal();

    __host__ __device__ float axis_min(int) const;
    __host__ __device__ float axis_max(int) const;

    __host__ __device__ Ray get_reflected_ray(const vec3& in, const point3& hit_point, const float power, const float coeff) const;
    __host__ __device__ Ray get_transmitted_ray(const vec3& in, const point3& hit_point, const float power, const float coeff) const;
};