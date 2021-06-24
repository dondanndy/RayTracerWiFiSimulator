#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include "parameters.h"
#include "wall.h"
#include "ray.h"
#include "receptor.h"
#include "read_map.h"
#include "cudaStructs/cuda3DMatrix.h"

// #ifdef BUILD_CUDA
// #include "kernel.cuh"
// #endif

// #ifdef BUILD_SERIAL
#include "serial_simulation.h"
// #endif

void test() {
    //Wall wall_test = Wall(point3(0, -1, -1), point3(0, 1, 1), point3(0, 1, -1), point3(0, -1, 1), 1);
    Wall wall_test = Wall(point3(3, -3, 0), point3(3, 3, 0), point3(-3, -3, 0), point3(-3, 3, 0), 1);
    Receptor rep_test = Receptor(point3(0, 0, 0), 1);

    //Ray ray_test = Ray(point3(4,4,0), point3(-1, -1, 0), 1);
    //Ray ray_test = Ray(point3(-4, -4, 0), point3(1, 1, 0), 1);
    //Ray ray_test = Ray(point3(0, -3, 0), point3(0, 1, 0), 1);
    //Ray ray_test = Ray(point3(1, 3, 0), point3(0, -1, 0), 1);
    
    //Ray ray_test = Ray(point3(1.5, 1.5, -3.0 / std::sqrt(2)), vec3(-1 / std::sqrt(2), -1 / std::sqrt(2), 1), 1);
    //Ray ray_test = Ray(point3(3.0 / std::sqrt(2), 3.0 / std::sqrt(2), 0), point3(-1, -1, 0), 1);

    //Ray ray_test = Ray(point3(3, -3, 0), vec3(-1, 1, 0), 1);
    //Ray ray_test = Ray(point3(0, -3, 3), vec3(0, 1, -1), 1);
    Ray ray_test = Ray(point3(0, 1, 0), vec3(0, -1, -1), 1);


    float dist, pow;
    Ray reflected_ray, transmitted_ray;
    if (wall_test.hit(ray_test, dist, pow)) {
        //std::cout << "Wall hit at " << dist << std::endl;
        printf("Wall hit at %e\n", dist);

        wall_test.get_hit_rays(ray_test.direction(), ray_test.at(dist), pow, reflected_ray, transmitted_ray);
        // printf("Reflected power = %e\n", reflected_ray.power());
    //        printf("Transmitted power = %e\n", transmitted_ray.power());
    }
    else {
        std::cout << "No wall hit :(\n";
    }

    //cuda3DMatrix<float> data_prop(181, 360, 1);

    //float* data = (float*)malloc(data_prop.size() * sizeof(float));
    //read_antenna_pattern("../../antenna_patterns/cilindrical_pattern.txt", data);
    //
    //float dist, pow;
    //if (rep_test.hit(ray_test, dist, pow)) {
    //    std::cout << "Sphere hit at dist = " << dist << std::endl;

    //    vec3 normal = rep_test.get_normal(ray_test.at(dist));

    //    //printf("Sphere normal is (%.3f, %.3f,%.3f)\n", normal.x(), normal.y(), normal.z());
    //    printf("Hit point is (%.3f, %.3f,%.3f)\n", ray_test.at(dist).x(), ray_test.at(dist).y(), ray_test.at(dist).z());


    //    //std::cout << "Sphere normal = " << normal.x() << << std::endl;

    //    std::cout << "Azimut = " << rep_test.hit_azimut(ray_test)* 180 / PI << std::endl;
    //    std::cout << "Elevation = " << rep_test.hit_elevation(ray_test) * 180 / PI << std::endl;

    //    //std::cout << "Directivity = " << interpolate_directivity(data, data_prop, 70, 10) << std::endl;
    //    std::cout << "Directivity = " << interpolate_directivity(data, data_prop, rep_test.hit_azimut(ray_test) * 180 / PI, rep_test.hit_elevation(ray_test) * 180 / PI) << std::endl;
    //}
    //else {
    //    std::cout << "No sphere hit :(\n";
    //}

    //free(data);

}

//__global__ void test_fill(float* tot, cuda3DMatrix<float> test) {
//    fill_block_data(2.0, tot, test);
//}
//
//__global__ void test_acc(float* tot, float* threads, cuda3DMatrix<float> threads_prop) {
//    fill_block_data(0.0, tot, cuda3DMatrix<float>(22, 11, 1));
//
//    accumulate_block_data(tot, threads, threads_prop);
//}

bool read_antenna_pattern(const std::string& filename, float* data) {
    //Read from file generated in MATLAB.
    //IMPORTANT: Assumes angles in 0:359 for azimut and 90:90 for elevation, so it will import a 181x360 matrix.

    std::ifstream file(filename);
    if (!file) {
        std::cout << "Error reading map file!" << std::endl;
        return false;
    }

    std::string line, val;
    size_t i = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);

        while (std::getline(ss, val, ',')) {
            data[i] = std::atof(val.c_str());
            i++;
        }
    }

    return true;
}

std::vector<Receptor> place_receptors(float size) {

    std::vector<Receptor> receps;
    double z = 0.6;

    for (int i = -2; i < 3; i++) {
        for (int j = -3; j < 4; j++) {
            receps.emplace_back(point3(i, j, z), size);
        }
    }

    return receps;
}

bool write_to_file(const std::string& filename, float* recep_data, std::vector<Receptor> receptors) {
    std::FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) return false;

    for (size_t i = 0; i < receptors.size(); i++) {
        fprintf(f, "%f,%f,%f,%e\n", receptors[i].get_origin().x(), receptors[i].get_origin().y(), receptors[i].get_origin().z(), recep_data[i]);
    }

    fclose(f);

    return true;
}

int main(){

    //Map
    //std::vector<Wall> map = read_map("../../maps/lab_map_2tablas.txt");
    //std::vector<Wall> map = read_map("../../maps/lab_map_1tabla.txt");
    std::vector<Wall> map = read_map("../../maps/lab_map.txt");

    //Receptors
    //TODO: Include in map??
    std::vector<Receptor> receptors = place_receptors(RECEPTOR_RADIUS);

    //Receptor directivity
    cuda3DMatrix<float> directivity_prop(181, 360, 1);
    float* directivity = (float*)malloc(directivity_prop.size());
    if (!directivity) {
        std::cout << "Error in directivity malloc!" << std::endl;
    }

    if (!read_antenna_pattern("../../antenna_patterns/cilindrical_pattern.txt", directivity)) {
        std::cout << "Error while reading antenna directivity!" << std::endl;
    }

    //Data for results.
    float* results = (float*)malloc(receptors.size() * sizeof(float));

    // #ifdef BUILD_CUDA
    //     cudaDeviceProp props;
    //     cudaGetDeviceProperties(&props, 0);

    //     std::cout << props.name << std::endl;
    //     std::cout << props.sharedMemPerBlock / 1024 << "KB" << std::endl;
    //     std::cout << props.kernelExecTimeoutEnabled << std::endl;

    //     setup_and_run_CUDA_simulation(map, receptors, directivity, directivity_prop, results);    
    // #endif

    // #ifdef BUILD_SERIAL
    run_serial_simulation(map, receptors, directivity, directivity_prop, results);
    // #endif
    
    //Log results.
    //write_to_file("2tablas.txt", results, receptors);
    //write_to_file("1tabla.txt", results, receptors);
    write_to_file("vacio.txt", results, receptors);

    free(results);
    free(directivity);

    //CLEANUP ------------------------------------------------
    ////Bloques que necesitamos
    //dim3 blocks(1,1);
    //dim3 threads(5,5);

    //cuda3DMatrix<float> data_prop(22, 11, 5);

    //float* c_data = (float*)malloc(data_prop.size() * sizeof(float));

    //for (size_t i = 0; i < data_prop.x(); i++) {
    //    for (size_t j = 0; j < data_prop.y(); j++) {
    //        for (size_t k = 0; k < data_prop.z(); k++) {
    //            data_prop.at(c_data, i, j, k) = k + 1;
    //        }
    //    }
    //}

    //float* data;
    //alloc_and_copy_to_gpu(c_data, &data, data_prop);

    //cuda3DMatrix<float> tot_prop(22, 11, 1);
    //float* tot;
    //cudaMalloc((void**)&tot, tot_prop.size() * (size_t)sizeof(float));

    //test_acc << <blocks, threads >> > (tot, data, data_prop);

    //float* tot_data = (float*)malloc(tot_prop.size() * sizeof(float));
    //copy_from_gpu<float>(tot_data, tot, tot_prop);

    //for (size_t i = 0; i < tot_prop.x(); i++) {
    //    for (size_t j = 0; j < tot_prop.y(); j++) {
    //        std::cout << tot_prop.at(tot_data, i, j) << "\t";
    //    }
    //    std::cout << "\n";
    //}
    return 0;
}