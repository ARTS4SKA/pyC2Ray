#pragma once

namespace asora {

    size_t global_pass(
        double* xh, double* xh_av, double* xh_int, const double* temp,
        const double* ndens, const double* phi_ion, const double* clump, double dt,
        double bh00, double albpow, double colh0, double temph0, double abu_c,
        size_t n_cells, size_t block_size
    );

}  // namespace asora
