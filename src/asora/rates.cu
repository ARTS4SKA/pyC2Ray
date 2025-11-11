#include "rates.cuh"

namespace {

    // Limit to consider a cell "optically thin/thick"
    constexpr double tau_photo_limit = 1.e-7;

    // Reference ionizing flux (strength of source is given in this unit)
    constexpr double s_star_ref = 1e48;

    // Utility function to look up the integral value corresponding to an
    // optical depth τ by doing linear interpolation.
    __device__ double photo_lookuptable(
        const double *table, double tau, double minlogtau, double dlogtau, int num_tau
    );

}  // namespace

namespace asora {

    // ========================================================================
    // Compute photoionization rate from in/out column density by looking up
    // values of the integral ∫L_v*e^(-τ_v)/hv in precalculated tables. These
    // tables are assumed to have been copied to device memory in advance using
    // photo_table_to_device()
    // ========================================================================
    __device__ double photoion_rates_gpu(
        double strength, double coldens_in, double coldens_out, double Vfact,
        double sig, const double *photo_thin_table, const double *photo_thick_table,
        double minlogtau, double dlogtau, int num_tau
    ) {
        // Compute optical depth and ionization rate depending on whether the cell is
        // optically thick or thin
        auto tau_in = coldens_in * sig;
        auto tau_out = coldens_out * sig;
        strength /= Vfact;

        // PH (08.10.23) I'm confused about the way the rates are calculated
        // differently for thin/thick cells. The following is taken verbatim from
        // radiation_photoionrates.F90 lines 276 - 303 but without true
        // understanding... Names are slightly different to simpify notatio

        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return strength * (tau_out - tau_in) *
                   photo_lookuptable(
                       photo_thin_table, tau_out, minlogtau, dlogtau, num_tau
                   );

        auto phi_photo_in =
            photo_lookuptable(photo_thick_table, tau_in, minlogtau, dlogtau, num_tau);
        auto phi_photo_out =
            photo_lookuptable(photo_thick_table, tau_out, minlogtau, dlogtau, num_tau);
        return strength * (phi_photo_in - phi_photo_out);
    }

    // ========================================================================
    // Grey-opacity test case photoionization rate, computed from analytical
    // expression rather than using tables. To use this version, compile
    // with the -DGREY_NOTABLES flag
    // ========================================================================
    __device__ double photoion_rates_test_gpu(
        double strength, double coldens_in, double coldens_out, double Vfact, double sig
    ) {
        // Compute optical depth and ionization rate depending on whether the cell
        // is optically thick or thin
        auto tau_in = coldens_in * sig;
        auto tau_out = coldens_out * sig;
        strength /= Vfact;

        // If cell is optically thin
        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return (strength * s_star_ref) * exp(-tau_in) * (tau_out - tau_in);

        // If cell is optically thick
        return (strength * s_star_ref) * (exp(-tau_in) - exp(-tau_out));
    }

}  // namespace asora

namespace {

    __device__ double photo_lookuptable(
        const double *table, double tau, double minlogtau, double dlogtau, int num_tau
    ) {
        // Find table index and do linear interpolation
        // Recall that tau(0) = 0 and tau(1:num_tau) ~ logspace(minlogtau,maxlogtau)
        // (so in reality the table has size num_tau+1)
        auto logtau = log10(max(1.0e-20, tau));
        auto real_i =
            min(double(num_tau), max(0.0, 1.0 + (logtau - minlogtau) / dlogtau));
        auto i0 = int(real_i);
        auto i1 = min(num_tau, i0 + 1);
        auto residual = real_i - double(i0);
        return table[i0] + residual * (table[i1] - table[i0]);
    }

}  // namespace
