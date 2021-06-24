#include "box.h"

__host__ __device__ box::box(const point3& og, const float side)
	: origin(og), side(side)
{
	const float half_side = side / 2;

	// Izquierda
	walls[0] = Wall(point3(og.x() - half_side, og.y() + half_side, og.z() - half_side),
					point3(og.x() - half_side, og.y() - half_side, og.z() - half_side),
					point3(og.x() - half_side, og.y() + half_side, og.z() + half_side),
					point3(og.x() - half_side, og.y() - half_side, og.z() + half_side), -1);

	//Superior
	walls[1] = Wall(point3(og.x() + half_side, og.y() + half_side, og.z() - half_side),
					point3(og.x() - half_side, og.y() + half_side, og.z() - half_side),
					point3(og.x() + half_side, og.y() + half_side, og.z() + half_side),
					point3(og.x() - half_side, og.y() + half_side, og.z() + half_side), -1);

	// Derecha
	walls[2] = Wall(point3(og.x() + half_side, og.y() - half_side, og.z() - half_side),
					point3(og.x() + half_side, og.y() + half_side, og.z() - half_side),
					point3(og.x() + half_side, og.y() - half_side, og.z() + half_side),
					point3(og.x() + half_side, og.y() + half_side, og.z() + half_side), -1);

	// Inferior
	walls[3] = Wall(point3(og.x() - half_side, og.y() - half_side, og.z() - half_side),
					point3(og.x() + half_side, og.y() - half_side, og.z() - half_side),
					point3(og.x() - half_side, og.y() - half_side, og.z() + half_side),
					point3(og.x() + half_side, og.y() - half_side, og.z() + half_side), -1);

	// Tejado
	walls[4] = Wall(point3(og.x() - half_side, og.y() - half_side, og.z() + half_side),
					point3(og.x() + half_side, og.y() - half_side, og.z() + half_side),
					point3(og.x() - half_side, og.y() + half_side, og.z() + half_side),
					point3(og.x() + half_side, og.y() + half_side, og.z() + half_side), -1);

	// Suelo
	walls[5] = Wall(point3(og.x() + half_side, og.y() - half_side, og.z() - half_side),
					point3(og.x() - half_side, og.y() - half_side, og.z() - half_side),
					point3(og.x() + half_side, og.y() + half_side, og.z() - half_side),
					point3(og.x() - half_side, og.y() + half_side, og.z() - half_side), -1);
}

__host__ __device__ bool box::hit(const Ray& r, float& final_power) const {
	float dist; // No lo usaremos.
	for (size_t i = 0; i < 6; i++) {
		if (walls[i].hit(r, dist, final_power)) {
			return true;
		}
	}

	return false;
}