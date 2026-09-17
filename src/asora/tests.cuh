#pragma once

#include "raytracing_lut.cuh"

#include <array>
#include <vector>

namespace asoratest {

    void cell_interpolator(
        double *coldens_data, double *dens_data, const std::array<size_t, 3> &out_shape
    );

    void geometric_factors(
        double *factors_data, const std::array<size_t, 3> &out_shape
    );

    void path_in_cell(double *path_data, const std::array<size_t, 3> &out_shape);

};  // namespace asoratest
