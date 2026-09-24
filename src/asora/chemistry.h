#pragma once

/* @file chemistry.h
 * @brief Global pass routine for the chemistry ODE solver.
 */

namespace asora {

    /* @brief Perform a global pass of the chemistry solver.
     *
     * The initial fraction, temperature and clumping fields are resident and are
     * uploaded once per timestep. The average fraction is resident too and is
     * updated in place; use average_fraction_to_host to read it back.
     *
     * @param xh_int Intermediate HI fraction (output)
     * @param phi_ion Photo-ionization rate (input)
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
        double* xh_int, const double* phi_ion, double dt, double bh00, double albpow,
        double colh0, double temph0, double abu_c, size_t n_cells, size_t block_size
    );

}  // namespace asora
