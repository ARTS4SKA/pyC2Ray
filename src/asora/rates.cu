#include "rates.cuh"

namespace {

    // Utility function to look up the integral value corresponding to an
    // optical depth τ by doing linear interpolation.
    // Recall that tau(0) = 0 and tau(1:num_tau) ~ logspace(minlogtau,maxlogtau)
    // (so in reality the table has size num_tau+1)
    __device__ double photo_lookuptable(
        const double *table, double tau, const asora::linspace<double> &logtau
    ) {
        // Clamp the log(tau) to be within the table range
        // (tau values below the minimum are set to the minimum)
        auto ltau = max(logtau.start, log10(tau));

        // Map ltau to its position in the table
        auto interp =
            min(static_cast<double>(logtau.num),
                1.0 + (ltau - logtau.start) / logtau.step);

        // Split the continuous index into integer and fractional parts
        // integral = floor of the index, used for table lookup
        // residual = fractional part, used for interpolation weight
        double integral;
        auto residual = modf(interp, &integral);

        // Determine the two table indices for linear interpolation and perform the
        // interpolation
        auto i0 = static_cast<size_t>(integral);
        auto i1 = min(logtau.num, i0 + 1);

        return (1 - residual) * table[i0] + residual * table[i1];
    }

}  // namespace

namespace asora {

    // Compute photoionization rate from in/out column density by looking up
    // values of the integral ∫L_v*e^(-τ_v)/hv in precalculated tables.
    __device__ double photo_rates_gpu(
        double tau_in, double tau_out, const photo_tables &tables,
        const linspace<double> &logtau
    ) {
        // Check if the cell is optically thin - simplified calculation
        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return (tau_out - tau_in) * photo_lookuptable(tables.thin, tau_out, logtau);

        // Cell is optically thick - use both tables
        auto phi_photo_in = photo_lookuptable(tables.thick, tau_in, logtau);
        auto phi_photo_out = photo_lookuptable(tables.thick, tau_out, logtau);
        return phi_photo_in - phi_photo_out;
    }

#ifdef GREY_NOTABLES
    __device__ double photo_rates_test_gpu(double tau_in, double tau_out) {
        // Check if cell is optically thin - linear approximation exp(x) ≈ 1 - x
        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return s_star_ref * exp(-tau_in) * (tau_out - tau_in);

        // Check if cell is optically thick - exponential formula
        return s_star_ref * (exp(-tau_in) - exp(-tau_out));
    }
#endif  // GREY_NOTABLES

}  // namespace asora
