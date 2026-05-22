#include "rates.cuh"

namespace asora {

    __host__ __device__ cuda::std::tuple<size_t, size_t, double> log_table_index(
        double x, const asora::linspace<double> &logscale
    ) {
        // Clamp the log(tau) to be within the table range
        // (tau values below the minimum are set to the minimum)
        auto lx = max(logscale.start, log10(x));

        // Map lx to its position in the table
        auto interp =
            min(static_cast<double>(logscale.num),
                1.0 + (lx - logscale.start) / logscale.step);

        // Split the continuous index into integer and fractional parts
        // integral = floor of the index, used for table lookup
        // residual = fractional part, used for interpolation weight
        double integral;
        auto residual = modf(interp, &integral);

        // Determine the two table indices for linear interpolation and perform the
        // interpolation
        auto i0 = static_cast<size_t>(integral);
        auto i1 = min(logscale.num, i0 + 1);

        return {i0, i1, residual};
    }

    __host__ __device__ double log_table_lookup(
        double x, const double *table, const asora::linspace<double> &logscale
    ) {
        auto &&[i0, i1, p] = log_table_index(x, logscale);
        return (1 - p) * table[i0] + p * table[i1];
    }

    // Compute photoionization rate from in/out column density by looking up
    // values of the integral ∫L_v*e^(-τ_v)/hv in precalculated tables.
    __device__ double photo_rates_gpu(
        double tau_in, double tau_out, const photo_tables &tables,
        const linspace<double> &logtau
    ) {
        // Check if the cell is optically thin - simplified calculation
        if (abs(tau_out - tau_in) <= tau_photo_limit)
            return (tau_out - tau_in) * log_table_lookup(tau_out, tables.thin, logtau);

        // Cell is optically thick - use both tables
        auto phi_photo_in = log_table_lookup(tau_in, tables.thick, logtau);
        auto phi_photo_out = log_table_lookup(tau_out, tables.thick, logtau);
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
