#pragma once

#include <array>

namespace asoratest {

    void cinterp_gpu(
        double *coldens_data, double *dens_data, const std::array<size_t, 3> &out_shape
    );

    void path_in_cell(double *path_data, const std::array<size_t, 3> &out_shape);

    std::array<int, 3> linthrd2cart(int q, int s);
    std::array<int, 2> cart2linthrd(int i, int j, int k);

};  // namespace asoratest
