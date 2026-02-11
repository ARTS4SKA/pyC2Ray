#include "rates.cuh"

namespace {

    // Limit to consider a cell "optically thin/thick"
    constexpr double tau_photo_limit = 1.e-7;

    // Utility function to look up the integral value corresponding to an
    // optical depth τ by doing linear interpolation.
    __device__ double photo_lookuptable(
        const double *table, double tau, const asora::linspace<double> &logtau
    ) {
        // Find table index and do linear interpolation
        // Recall that tau(0) = 0 and tau(1:num_tau) ~ logspace(minlogtau,maxlogtau)
        // (so in reality the table has size num_tau+1)
        auto ltau = max(logtau.start, log10(tau));

        auto interp =
            min(static_cast<double>(logtau.num),
                1.0 + (ltau - logtau.start) / logtau.step);

        double integral;
        auto residual = modf(interp, &integral);

        auto i0 = static_cast<size_t>(integral);
        auto i1 = min(logtau.num, i0 + 1);

        return (1 - residual) * table[i0] + residual * table[i1];
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
        double tau_in, double tau_out, const photo_tables &tables,
        const linspace<double> &logtau
    ) {
        // PH (08.10.23) I'm confused about the way the rates are calculated
        // differently for thin/thick cells. The following is taken verbatim from
        // radiation_photoionrates.F90 lines 276 - 303 but without true
        // understanding... Names are slightly different to simplify notation

        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return (tau_out - tau_in) * photo_lookuptable(tables.thin, tau_out, logtau);

        auto phi_photo_in = photo_lookuptable(tables.thick, tau_in, logtau);
        auto phi_photo_out = photo_lookuptable(tables.thick, tau_out, logtau);
        return phi_photo_in - phi_photo_out;
    }

#ifdef GREY_NOTABLES
    // ========================================================================
    // Grey-opacity test case photoionization rate, computed from analytical
    // expression rather than using tables.
    // ========================================================================
    __device__ double photo_rates_test_gpu(double tau_in, double tau_out) {
        // Reference ionizing flux (strength of source is given in this unit)
        constexpr double s_star_ref = 1e48;

        // If cell is optically thin
        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return s_star_ref * exp(-tau_in) * (tau_out - tau_in);

        // If cell is optically thick
        return s_star_ref * (exp(-tau_in) - exp(-tau_out));
    }
#endif  // GREY_NOTABLES

}  // namespace asora
