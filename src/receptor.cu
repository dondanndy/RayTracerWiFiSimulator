#include "receptor.h"

bool Receptor::hit(const Ray& r, float& distance, float& final_power, float* data, cuda3DMatrix<float> data_prop) const {

	float dist;
	if (intersect(r, dist)) {
		distance = dist;
		final_power = r.get_decay(dist) * interpolate_directivity(data, data_prop, hit_azimut(r), hit_elevation(r));
		return true;
	}
	else {
		return false;
	}
}

bool Receptor::intersect(const Ray& r, float& distance) const {
	float det, dist;
	det = (dot(r.direction(), r.origin() - this->origin));
	det = det * det;
	det = det - (r.origin() - this->origin).length_squared() + radius * radius;

	//At least two intersection points.
	if (det > 0) {
		dist = -(dot(r.direction(), r.origin() - this->origin)) - std::sqrt(det);
	}
	else {
		//No intersection.
		return false;
	}

	if (dist > 0){
		//Intersection point is in the directon of the ray.
		distance = dist;
		return true;
	}
	else {
		//Intersetion point is in the oposite direction, so no intersection.
		return false;
	}
}

vec3 Receptor::get_normal(const point3& point) const {
	return unit_vector(point - origin);
}

float Receptor::hit_azimut(const Ray& r) const {
	//We need it in [0:2Pi), from https://stackoverflow.com/a/25725005
	float ang = fmodf(std::atan2(-r.direction().y(), -r.direction().x()) + (2 * PI), 2 * PI)*180/PI;
	
	return ang - azim_offset;
}

float Receptor::hit_elevation(const Ray& r) const {
	float ang = std::acos(-r.direction().z()) * 180 / PI;

	return ang - elev_offset;
}