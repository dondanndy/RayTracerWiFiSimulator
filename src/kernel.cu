#include "kernel.cuh"

void setup_and_run_CUDA_simulation(std::vector<Wall>& map, std::vector<Receptor>& receptors,
	                               float* antenna_pattern, cuda3DMatrix<float> antenna_pattern_props,
	                               float* results)
{

	//--------------------------------------------
	// DATA COPY
	//--------------------------------------------

	//Map
	Wall* g_map;
	if (!alloc_and_copy_to_gpu<Wall>(map.data(), &g_map, map.size())) {
		std::cout << "Error al copiar el mapa!\n";
		return;
	}

	//Receptors
	//TODO: Include in map??
	Receptor* g_receptors;
	if (!alloc_and_copy_to_gpu<Receptor>(receptors.data(), &g_receptors, receptors.size())) {
		std::cout << "Error al copiar los receptores!\n";
		return;
	}

	//Receptors directivity data
	float* g_directivity;
	alloc_and_copy_to_gpu<float>(antenna_pattern, &g_directivity, antenna_pattern_props);

	//--------------------------------------------
	// CUDA PARAMETERS
	//--------------------------------------------

	//Blocks and threads
	constexpr size_t ANGLE_LIMIT = 4; //Each block will have assigned a 4x4-degree angle division.

	dim3 blocks((360 + ANGLE_LIMIT - 1) / ANGLE_LIMIT, (180 + ANGLE_LIMIT - 1) / ANGLE_LIMIT);
	dim3 threads(ANGLE_LIMIT, ANGLE_LIMIT);

	size_t total_blocks = blocks.x * blocks.y;

	//Global data
	cuda3DMatrix<float> full_data_props(receptors.size(), MAX_REBOUND, total_blocks);
	float* g_full_data;
	
	if (alloc_in_gpu<float>(&g_full_data, full_data_props) != cudaSuccess) {
		std::cout << "Error al reservar memoria para todos los datos!\n";
		return;
	}

	//Memory for each block
	size_t block_mem =  MAX_REBOUND * receptors.size() * ANGLE_LIMIT * ANGLE_LIMIT * sizeof(float);

	//--------------------------------------------
	// KERNEL LAUNCH
	//--------------------------------------------
	std::cout << "Launching kernel!\n";
	point3 origin(-1.0, -4.0, 0.7);
	run_simulation <<<blocks, threads,  block_mem>>>(origin, g_map, map.size(),
		                                              g_receptors, map.size(),
		                                              g_full_data, full_data_props,
		                                              g_directivity, antenna_pattern_props, ANGLE_STEP);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "Error: kernel has failed. Code = " << err << std::endl;
		return;
	}
	std::cout << "Kernel ended!\n";

	//--------------------------------------------
	// DATA RETRIEVAL
	//--------------------------------------------

	//Results from each block
	float* full_data = (float*)malloc(full_data_props.size());
	if (copy_from_gpu<float>(full_data, g_full_data, full_data_props) != cudaSuccess) {
		std::cout << "Error copying results back!, Code = " << cudaGetLastError() << std::endl;
	}

	//Accumulated results from every block
	cuda3DMatrix<float> acc_data_props = full_data_props.slice_2D();
	float* acc_data = (float*)malloc(acc_data_props.size());

	//Accumulate 3D array into 2D array
	for (size_t i = 0; i < full_data_props.nrows(); i++) {
		for (size_t j = 0; j < full_data_props.ncols(); j++) {
			acc_data_props.at(acc_data, i, j) = 0.0;

			for (size_t k = 0; k < full_data_props.ndepth(); k++) {
				if (acc_data_props.at(acc_data, i, j) < full_data_props.at(full_data, i, j, k))
					acc_data_props.at(acc_data, i, j) = full_data_props.at(full_data, i, j, k);
			}
		}
	}

	//Accumulate results from every block, a 2D array into a 1D vector as the results of the simulation.
	for (size_t i = 0; i < acc_data_props.nrows(); i++) {
		results[i] = 0.0;
		for (size_t j = 0; j < acc_data_props.ncols(); j++) {
			results[i] += acc_data_props.at(acc_data, i, j);
		}
	}

	//Free everything;
	cudaFree(g_map);
	cudaFree(g_receptors);
	cudaFree(g_full_data);
	cudaFree(g_directivity);

	free(full_data);
	free(acc_data);
	return;
}



__global__ void run_simulation(point3 origin, Wall* map, size_t map_size, 
                               Receptor* tx, size_t tx_size,
	                           float* tx_power, cuda3DMatrix<float> tx_props,
	                           float* directivity, cuda3DMatrix<float> directivity_prop,
	                           float ang_step)
{
	//Main device Simulation loop.
	
	//Data for each thread.
	extern __shared__ float block_data[];
	cuda3DMatrix<float> block_data_props(tx_size, MAX_REBOUND, blockDim.x*blockDim.y);

	Ray emitter(origin, vec3(1, 0, 0), EMITTING_POWER);
	size_t thread_offset = (threadIdx.x + threadIdx.y*blockDim.x)*tx_props.nrows()*tx_props.ncols();

	//Main loop

	//Every block has a fixed range of angles to evaluate. If the number of total angles is higher 
	//than the available threads, each of them will run more than one ray.
	//We split ranges in 2D little blocks.
	float limx = ang_step * 4.0;
	while (limx <= 4.0) {
		
		float limy = ang_step * 4.0;
		while (limy <= 4.0) {
			float ang = (threadIdx.x * ang_step + blockIdx.x * 4.0) * PI / 180;
			float azm = (90 - (threadIdx.y * ang_step + blockIdx.y * 4.0)) * PI / 180;

			emitter.set_direction(vec3(std::cos(ang), std::sin(ang), std::sin(azm)));
			run_ray(emitter, map, map_size, tx, tx_size, block_data + thread_offset, block_data_props.slice_2D(), directivity, directivity_prop);

			__syncthreads();

			accumulate_block_data(tx_power, tx_props, block_data, block_data_props);

			__syncthreads();

			fill_3D_matrix(0.0, block_data, block_data_props);

			limy += ang_step * 4.0;
		}

		limx += ang_step * 4.0;
	}
}

__device__ void run_ray(Ray r, Wall* map, size_t map_size,
	                    Receptor* tx, size_t tx_size,
	                    float* tx_power, cuda3DMatrix<float> tx_power_prop,
	                    float* directivity, cuda3DMatrix<float> directivity_prop)
{
	cudaStack<cudaPair<Ray, unsigned int>> ray_stack(MAX_STACK_VALUES);

	cudaPair<Ray, unsigned int> ray(r, 0); //Ray to evaluate.

	//Main loop for each ray until every one of them has died.
	float dist, power;
	Ray reflected_ray, transmitted_ray;

	bool more_rays = true; //TODO:could this be better??
	do {
		size_t reb = ray.second(); //Number of rebounds that this ray has made.
		//Walls
		float hit_dist = INFINITY; //Distance
		size_t wall_hit; //Wall which has been hit.
		for (size_t j = 0; j < map_size; j++) {
			if (map[j].hit(ray.first(), dist, power)) {
				if (dist < hit_dist) {
					hit_dist = dist;
					wall_hit = j;
				}
			}
		}

		//Receptors.
		for (size_t j = 0; j < tx_size; j++) {
			if (tx[j].hit(ray.first(), dist, power, directivity, directivity_prop)) {
				if (power > tx_power_prop.at(tx_power, j, reb) && dist < hit_dist)
					tx_power_prop.at(tx_power, j, reb) = power;
			}
		}


		if (power > CUTOFF_POWER && reb < MAX_REBOUND) {
			//Evaluate reflected ray, save transmitted ray.

			map[wall_hit].get_hit_rays(ray.first().direction(), ray.first().at(hit_dist), power, reflected_ray, transmitted_ray);

			//TODO: smth similar to emplace_back.
			ray_stack.add(cudaPair<Ray, unsigned int>(transmitted_ray, ray.second() + 1));
			ray = cudaPair<Ray, unsigned int>(reflected_ray, ray.second() + 1);
		}
		else {
			//We will keep evaluating rays until the stack is empty.
			more_rays = ray_stack.pop(ray);
		}
	} while (more_rays);
}

__device__ void fill_3D_matrix(float val, float* data, cuda3DMatrix<float> data_props)
{
	//We slice the 3D array so every thread select a position on the 2D structure and set
	//every value on the z (depth) axis.

	size_t nx = 0;
	while (nx < data_props.nrows()) {
		size_t ny = 0;
		while (ny < data_props.ncols()) {
			if ((threadIdx.x + nx < data_props.nrows()) && (threadIdx.y + ny < data_props.ncols())) {
				for (size_t k = 0; k < data_props.ndepth(); k++) {
					data_props.at(data, threadIdx.y + ny, threadIdx.x + nx, k) = val;
				}
			}
			ny += blockDim.y;
		}

		nx += blockDim.x;
	}
}

__device__ void accumulate_block_data(float* full_data, cuda3DMatrix<float> full_data_prop,
	                                  float* block_data, cuda3DMatrix<float> block_data_prop)
{
	/*
		Here we brake down the full array in 2D slices that match the threads of the block.
		
		Then, every thread evaluates a cell in its depth(threads) accumulating the results on the 2D block_data array.
	*/
	
	size_t block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * full_data_prop.slice_2D().elements();

	size_t nx = 0; //Columns offset.
	while (nx < block_data_prop.nrows()) {
		size_t ny = 0; //Rows offset.
		while (ny < block_data_prop.ncols()) {
			if ((threadIdx.x + nx < block_data_prop.nrows()) && (threadIdx.y + ny < block_data_prop.ncols())) {

				//We now iterate over the threads of the block.
				for (size_t k = 0; k < block_data_prop.ndepth(); k++) {
					if (full_data_prop.at(full_data+block_offset, threadIdx.y + ny, threadIdx.x + nx) < block_data_prop.at(block_data, threadIdx.x + nx, threadIdx.y + ny, k))
						full_data_prop.at(full_data+block_offset, threadIdx.y + ny, threadIdx.x + nx) = block_data_prop.at(block_data, threadIdx.y + ny, threadIdx.x + nx, k);
				}
			}
			ny += blockDim.y;
		}
		nx += blockDim.x;
	}
}