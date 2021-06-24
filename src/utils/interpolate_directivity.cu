#include "interpolate_directivity.h"
#include "cuda_runtime.h"
#include <cmath>

__host__ __device__ float interpolate_directivity(float* data, cuda3DMatrix<float> data_prop, float azim, float elev) {
    //Bilinear 2D interpolation taken from
    //https://en.wikipedia.org/wiki/Bilinear_interpolation#Alternative_algorithm
    //NOTE:We had taken into account the fact that our "blocks" have length 1.
    size_t x1, x2, y1, y2; //Extremes of the block.
    float Q11, Q12, Q21, Q22; //Values at extremes
    float a0, a1, a2, a3;

    x1 = (size_t)floor(azim);
    x2 = (size_t)ceil(azim);

    y1 = (size_t)floor(elev);
    y2 = (size_t)ceil(elev);

    //We need some dimension on both axis.
    //This might be optimized somehow.
    if (x1 == x2) {
        x2++;
    }

    if (y1 == y2) {
        y2++;
    }

    Q11 = data_prop.at(data, y1, x1);
    Q12 = data_prop.at(data, y2, x1);
    Q21 = data_prop.at(data, y1, x2);
    Q22 = data_prop.at(data, y2, x2);

    a0 = Q11 * x2 * y2 - Q12 * x2 * y1 - Q21 * x1 * y2 + Q22 * x1 * y1;
    a1 = -Q11 * y2 + Q12 * y1 + Q21 * y2 - Q22 * y1;
    a2 = -Q11 * x2 + Q12 * x2 + Q21 * x1 - Q22 * x1;
    a3 = Q11 - Q12 - Q21 + Q22;

    return (a0 + a1 * azim + a2 * elev + a3 * azim * elev);
}