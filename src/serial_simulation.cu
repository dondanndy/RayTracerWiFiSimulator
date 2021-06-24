#include "serial_simulation.h"
#include "utils/interpolate_directivity.h"

void run_serial_simulation(const std::vector<Wall>& map, const std::vector<Receptor>& receptors,
	                       float* antenna_pattern, cuda3DMatrix<float> antenna_pattern_props,
	                       float* results)
{
	//Data to fill.
	cuda3DMatrix<float> data_prop(receptors.size(), MAX_REBOUND, 1);
	float* data = (float*)malloc(data_prop.size());
	if (!data) {
		std::cout << "Error at data malloc!" << std::endl;
		return;
	}

	//Init
	for (size_t i = 0; i < data_prop.nrows(); i++) {
		for (size_t j = 0; j < data_prop.ncols(); j++) {
			data_prop.at(data, i, j) = 0.0f;
		}
	}

	//Main loop.
	Ray emitter(point3(-1.0, -4.0, 0.6), vec3(1, 0, 0), EMITTING_POWER);

	for (float azim = 0; azim < 360; azim += ANGLE_STEP) {
		printf("Running azimut = %.4f\n", azim);

		for (float el = 0; el < 180; el += ANGLE_STEP) {
			//printf("Running elevation = %.4f\n", el);
			emitter.set_direction(vec3(std::cos(azim*PI/180.0) * std::sin(el*PI/180.0), std::sin(azim*PI/180.0) * std::sin(el*PI/180.0), std::cos(el*PI/180.0)));
			emitter.set_power(EMITTING_POWER * interpolate_directivity(antenna_pattern, antenna_pattern_props, azim, el));

			run_ray(emitter, map, receptors, data, data_prop, antenna_pattern, antenna_pattern_props);
		}
	}

	//Acumulate
	for (size_t i = 0; i < data_prop.nrows(); i++) {
		results[i] = 0.0f;
		for (size_t j = 0; j < data_prop.ncols(); j++) {
			results[i] += data_prop.at(data, i, j);
		}
	}

	free(data);
	return;
}

void run_ray(const Ray& r, const std::vector<Wall>& map, const std::vector<Receptor>& tx,
	         float* data, cuda3DMatrix<float> data_prop,
	         float* directivity, cuda3DMatrix<float> directivity_prop)
{

	cudaStack<cudaPair<Ray, unsigned int>> ray_stack(MAX_STACK_VALUES);

	cudaPair<Ray, unsigned int> ray(r, 0); //Ray to evaluate.

	//Main loop for each ray until every one of its children has died.
	float dist, power;
	Ray reflected_ray, transmitted_ray;

	bool more_rays = true; //TODO:could this be better??
	do {
		power = 0;
		size_t reb = ray.second(); //Number of rebounds that this ray has made.

		//Walls
		float hit_dist = INFINITY; //Distance
		int wall_hit = -1; //Wall which has been hit.
		for (size_t j = 0; j < map.size(); j++) {
			if (map[j].hit(ray.first(), dist, power)) {
				if (dist < hit_dist) {
					hit_dist = dist;
					wall_hit = j;
					//printf("dist = %.3f, power = %e\n", dist, power);
				}
			}
		}

		//Receptors.
		for (size_t j = 0; j < tx.size(); j++) {
			if (tx[j].hit(ray.first(), dist, power, directivity, directivity_prop)) {
				if (power > data_prop.at(data, j, reb) && dist < hit_dist) {
					data_prop.at(data, j, reb) = power;
					
					if (j == 16 && reb == 0) {
						printf("Recibido! wall_hit = %d, dir = (%.4f, %.4f, %.4f), pow = %e\n", wall_hit, ray.first().direction().x(), ray.first().direction().y(), ray.first().direction().z(), power);
					}
				}
			}
		}

		//We only continue evaluating the ray if:
		//   - Has enough power left
		//   - Has not reached the maximum number of rebounds.
		//   - Has hit any of the walls of the map.
		if (power > CUTOFF_POWER && reb < MAX_REBOUND && wall_hit > -1) {
			//Evaluate reflected ray, save transmitted ray.

			map[wall_hit].get_hit_rays(ray.first().direction(), ray.first().at(hit_dist), power, reflected_ray, transmitted_ray);

			//TODO: smth similar to emplace_back.
			ray_stack.add(cudaPair<Ray, unsigned int>(transmitted_ray, reb + 1));
			ray = cudaPair<Ray, unsigned int>(reflected_ray, reb + 1);
		}
		else {
			more_rays = ray_stack.pop(ray);
		}
	} while (more_rays);
}

