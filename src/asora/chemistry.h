#pragma once

/* @file chemistry.h
 * @brief Global pass routine for the chemistry ODE solver.
 */

namespace asora {

    /* @brief Perform a global pass of the chemistry solver.
     *
     * @param xh Initial HI fraction (input)
     * @param xh_av Average HI fraction (output)
     * @param xh_int Intermediate HI fraction (output)
     * @param temp Temperature field (input)
     * @param ndens Hydrogen density field (input)
     * @param phi_ion Photo-ionization rate (input)
     * @param clump Clumping factor field (input)
     * @param dt Time step size
     * @param bh00 Hydrogen recombination parameter (value at 10^4K)
     * @param albpow Hydrogen recombination parmaeter (power-law index)
     * @param colh0 Hydrogen collisional ionization parameter
     * @param temph0 Hydrogen ionization energy expressed in K
     * @param abu_c Carbon abundance
     * @param n_cells Number of cells in the simulation
     * @param block_size CUDA block size
     *
     * @return Number of converged cells
     */
    size_t global_pass(
        double* xh, double* xh_av, double* xh_int, const double* temp,
        const double* ndens, const double* phi_ion, const double* clump, double dt,
        double bh00, double albpow, double colh0, double temph0, double abu_c,
        size_t n_cells, size_t block_size
    );

}  // namespace asora
