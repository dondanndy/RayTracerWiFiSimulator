#include "Wall.h"
#include <limits>

__host__ __device__ void Wall::calc_normal() {
	vec3 vec1 = limits[1] - limits[0];
	vec3 vec2 = limits[2] - limits[0];

	normal = unit_vector(cross(vec1, vec2));
}

__host__ __device__ float Wall::axis_min(int ax) const {

	float min = INFINITY;

	for (int i = 0; i < 4; i++) {
		if (limits[i].e[ax-1] < min) {
			min = limits[i].e[ax-1];
		}
	}

	return min;
}

__host__ __device__ float Wall::axis_max(int ax) const {

	float max = -INFINITY;

	for (int i = 0; i < 4; i++) {
		if (limits[i].e[ax-1] > max) {
			max = limits[i].e[ax - 1];
		}
	}
	return max;
}

__host__ __device__ float Wall::intersect(const Ray& r) const {

	float denom = dot(r.direction(), normal);
	
	if (abs(denom) > 1e-6) {
		vec3 trasl_normal = limits[0] - r.origin();
		float d = dot(trasl_normal, normal) / denom;
		
		if (d > 1e-3) {
			//NOTE:Sometimes, due to floating point rounding errors, some hit points are not _exactly_ in the incidance plane,
			//so this returns absourdly small distances that propagates up until the power calculation giving extremely high values.
			//Consider that value an epsilon guard for those cases.
			return d;
		}
		else {
			return INFINITY;
		}
	}

	// El vector es paralelo al plano o casi.
	return INFINITY;
}

__host__ __device__ bool Wall::check_lims(const point3& p) const {
	bool check_x = false;
	bool check_y = false;
	bool check_z = false;

	float eps = 1e-4;

	if (p.x() >= axis_min(1)-eps && p.x() <= axis_max(1)+eps) {
		check_x = true;
	}

	if (p.y() >= axis_min(2)-eps && p.y() <= axis_max(2)+eps) {
		check_y = true;
	}

	if (p.z() >= axis_min(3)-eps && p.z() <= axis_max(3)+eps) {
		check_z = true;
	}

	return check_x && check_y && check_z;
}

__host__ __device__ bool Wall::hit(const Ray& r, float& distance, float& final_power) const {

	float dist = intersect(r);

	if (dist == INFINITY || !check_lims(r.at(dist))) {
		// El rayo no golpea la pared.
		distance = -1;
		final_power = -1;
		return false;
	}

	distance = dist;
	final_power = r.get_decay(dist);
	return true;


	/*if (check_lims(r.at(dist))) {
		distance = dist;
		final_power = get_decay(r.power, distance);
		return true;
	}
	else
	{
		distance = -1;
		final_power = -1;
		return false;
	}*/
}

__host__ __device__ void Wall::get_hit_rays(const vec3& in, const point3& hit_point, const float power, Ray& reflected, Ray& transmitted) const {
	//Reflection coefficient
	float cos_ang = std::abs(dot(-in, normal));

	//s component
	float sqrt_fac = std::sqrtf(permitivity - (1.0f - cos_ang*cos_ang));

	float refl_coeff_s = (cos_ang - sqrt_fac) / (cos_ang + sqrt_fac);
	refl_coeff_s = refl_coeff_s * refl_coeff_s;

	//p component
	float refl_coeff_p = (permitivity*cos_ang - sqrt_fac) / (permitivity * cos_ang + sqrt_fac);
	refl_coeff_p = refl_coeff_p * refl_coeff_p;

	//Assuming no polarization
	float refl_coeff = 0.5f * (refl_coeff_s + refl_coeff_p);

	//We can now get both rays.
	reflected = get_reflected_ray(in, hit_point, power, refl_coeff);
	transmitted = get_transmitted_ray(in, hit_point, power, 1.0f - refl_coeff);
}

__host__ __device__ Ray Wall::get_reflected_ray(const vec3& in, const point3& hit_point, const float power, const float coeff) const {
	return Ray(hit_point, 2 * dot(normal, -in) * normal + in, coeff*power);
}

__host__ __device__ Ray Wall::get_transmitted_ray(const vec3& in, const point3& hit_point, const float power, const float coeff) const {
	return Ray(hit_point, in, coeff * power);
}