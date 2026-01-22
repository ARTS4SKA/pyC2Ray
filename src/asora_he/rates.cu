#include "rates.cuh"

namespace {

    // Limit to consider a cell "optically thin/thick"
    constexpr double tau_photo_limit = 1.e-7;

    // Reference ionizing flux (strength of source is given in this unit)
    [[maybe_unused]] constexpr double s_star_ref = 1e48;

    // Utility function to look up the integral value corresponding to an
    // optical depth τ by doing linear interpolation.
    __device__ double photo_lookuptable(
        const double *table, int nf, double tau, const asora::linspace<double> &logtau
    ) {
        int i0, i1;
        double real_i;

        // Find table index and do linear interpolation
        // Recall that tau(0) = 0 and tau(1:NumTau) ~ logspace(logtau.start,maxlogtau)
        // (so in reality the table has size NumTau+1)
        // Find table index and weight for linear interpolation
        if (log10(tau) < logtau.start) {
            real_i = 0.0;
            i0 = 0;
            i1 = 1;
        } else if (log10(tau) <= logtau.stop()) {
            // logtau.num - 1?
            real_i = 1.0 + (log10(tau) - logtau.start) / logtau.step;
            i0 = int(floor(real_i));
            i1 = int(ceil(real_i));
        } else {
            real_i = float(logtau.num);
            i0 = logtau.num;
            i1 = logtau.num + 1;
        }
        double w1 = real_i - float(i0);
        double w0 = float(i1) - real_i;

        // MB (02.10.204): Look for the table value. In the Helium update the tables
        // are a 2D array with shape (N_tau, N_freq) that is table.T.ravel() before
        // being passed to C++ routine.
        table += nf * (logtau.num + 1);
        return table[i0] * w0 + table[i1] * w1;
    }

}  // namespace

namespace asora {

    // ========================================================================
    // Compute photoionization rate from in/out column density by looking up
    // values of the integral ∫L_v*e^(-τ_v)/hv in precalculated tables. These
    // tables are assumed to have been copied to device memory in advance using
    // photo_table_to_device()
    // ========================================================================
    __device__ double photo_rates_gpu(
        double tau_in, double tau_out, int nf, const photo_tables &ion_tables,
        const linspace<double> &logtau
    ) {
        // MB (23.09.24): Rather then re-calculating the tau_in and tau_out as in the
        // HI only raytracing. Here, we pass these two variables (TODO: this could be
        // implemented also in the HI only).

        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return (tau_out - tau_in) *
                   photo_lookuptable(ion_tables.thin, nf, tau_out, logtau);

        double phi_photo_in = photo_lookuptable(ion_tables.thick, nf, tau_in, logtau);
        double phi_photo_out = photo_lookuptable(ion_tables.thick, nf, tau_out, logtau);
        return phi_photo_in - phi_photo_out;
    }

    // ========================================================================
    // Grey-opacity test case photoionization rate, computed from analytical
    // expression rather than using tables. To use this version, compile with the
    // -DGREY_NOTABLES flag
    // ========================================================================
    __device__ double photo_rates_test_gpu(double tau_in, double tau_out) {
        // If cell is optically thin
        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return s_star_ref * (tau_out - tau_in) * exp(-tau_in);

        // If cell is optically thick
        return s_star_ref * (exp(-tau_in) - exp(-tau_out));
    }

}  // namespace asora
