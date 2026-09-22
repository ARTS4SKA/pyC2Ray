#include "raytracing.cuh"

#include "../asora/memory.h"
#include "../asora/utils.cuh"

#include <cuda_runtime.h>

#include <cuda/std/cassert>
#include <cuda/std/span>
#include <exception>

namespace asora {

    __device__ void element_data::partition_column_density(int q) {
        /// Partition the column density array into 3 shared memory banks for easier
        /// interpolation
        shared_cdens = {
            column_density + asora::cells_to_shell(q - 2),
            column_density + asora::cells_to_shell(q - 3),
            column_density + asora::cells_to_shell(q - 4)
        };
    }

    __device__ double3 density_maps::get(size_t index) const {
        auto np = ndens[index];
        auto nHI = np * abu_h * (1.0 - xHII[index]);
        auto nHeI = np * abu_he * (1.0 - xHeII[index] - xHeIII[index]);
        auto nHeII = np * abu_he * xHeII[index];

        return {nHI, nHeI, nHeII};
    }

}  // namespace asora

namespace {

    using namespace asora;

    element_data make_element_data(
        asora::buffer_tag ion, asora::buffer_tag cdens, asora::buffer_tag sigma
    ) {
        return {
            get_data_view<double>(ion),
            get_data_view<double>(cdens),
            get_data_view<double>(sigma),
        };
    }

    density_maps make_density_maps() {
        return {
            get_data_view<double>(buffer_tag::number_density),
            get_data_view<double>(buffer_tag::fraction_HII),
            get_data_view<double>(buffer_tag::fraction_HeII),
            get_data_view<double>(buffer_tag::fraction_HeIII)
        };
    }

    struct optical_depth {
        /// Optical depth at the start of a cell.
        double in = 0.0;

        /// Optical depth at the end of a cell.
        double out = 0.0;

        /// Optical depth of the cell itself (difference between out and in).
        __device__ double cell() const { return out - in; }
    };

    struct photo_rates {
        double ionization_HI = 0.0;
        double ionization_HeI = 0.0;
        double ionization_HeII = 0.0;
        double heating = 0.0;

        __device__ photo_rates &operator+=(const photo_rates &other) {
            ionization_HI += other.ionization_HI;
            ionization_HeI += other.ionization_HeI;
            ionization_HeII += other.ionization_HeII;
            heating += other.heating;
            return *this;
        }
    };

    // Fit factors adapted from Ricotti et al (2002).
    __device__ cuda::std::array<double3, 2> fit_factors(double xHII) {
        constexpr double3 CR1 = {0.3908, 0.0554, 1.0};
        constexpr double3 bR1 = {0.4092, 0.4614, 0.2663};
        constexpr double3 dR1 = {1.7592, 1.6660, 1.3163};
        constexpr double3 CR2 = {0.6941, 0.0984, 3.9811};
        constexpr double3 aR2 = {0.2, 0.2, 0.4};
        constexpr double3 bR2 = {0.38, 0.38, 0.34};

        double3 y1R{
            CR1.x * pow(1.0 - pow(xHII, bR1.x), dR1.x),
            CR1.y * pow(1.0 - pow(xHII, bR1.y), dR1.y),
            CR1.z * pow(1.0 - pow(xHII, bR1.z), dR1.z)
        };
        double3 y2R{
            CR2.x * pow(xHII, aR2.x) * pow(1.0 - pow(xHII, bR2.x), 2),
            CR2.y * pow(xHII, aR2.y) * pow(1.0 - pow(xHII, bR2.y), 2),
            CR2.z * pow(xHII, aR2.z) * pow(1.0 - pow(xHII, bR2.z), 2)
        };
        return {y1R, y2R};
    }

    // Compute the secondary ionization and heating rates based on the primary
    // photo-heating rates. Return the secondary ionization rates for HI, HeI, and the
    // heating rate
    __device__ photo_rates compute_secondary_ionization(
        const double *__restrict__ heat_factors, const double3 &heat_rates,
        const double3 &y1R, const double3 &y2R
    ) {
        // Ionization energies in erg, converted from eV.
        constexpr double ev2erg = 1.6021766339999e-12;
        constexpr double ion_HI = 13.598 * ev2erg;
        constexpr double ion_HeI = 24.587 * ev2erg;

        using span = cuda::std::span<const double, 3>;
        span f1ion(heat_factors, 3);
        span f2ion(heat_factors + 3, 3);
        span f1heat(heat_factors + 6, 3);
        span f2heat(heat_factors + 9, 3);

        auto make_factor = [&](const span &f) {
            return heat_rates.x * f[0] + heat_rates.y * f[1] + heat_rates.z * f[2];
        };

        // Compute secondary ionization terms:
        auto sum_f1ion = make_factor(f1ion);
        auto sum_f2ion = make_factor(f2ion);
        auto sum_f1heat = make_factor(f1heat);
        auto sum_f2heat = make_factor(f2heat);

        auto df_ion_HI = (y1R.x * sum_f1ion + y2R.x * sum_f2ion) / ion_HI;
        auto df_ion_HeI = (y1R.y * sum_f1ion + y2R.y * sum_f2ion) / ion_HeI;
        auto df_heat = heat_rates.x + heat_rates.y + heat_rates.z - y1R.z * sum_f1heat +
                       y2R.z * sum_f2heat;

        return {df_ion_HI, df_ion_HeI, 0.0, df_heat};
    }

    // Compute the photoionization and heating rates for a given cell based on the
    // incoming and outgoing column densities, cross sections, and pre-computed
    // photoionization tables.
    __device__ photo_rates compute_photo_rates(
        const double3 &cd_in, const double3 &cd_out,
        const double *__restrict__ cross_section_HI,
        const double *__restrict__ cross_section_HeI,
        const double *__restrict__ cross_section_HeII,
        const double *__restrict__ heat_factors, const double3 &y1R, const double3 &y2R,
        const photo_tables<> &__restrict__ ion_tables,
        const photo_tables<> &__restrict__ heat_tables, const linspace<> &logtau,
        size_t num_freq
    ) {
        photo_rates rates;

        // Frequency loop.
        for (size_t nf = 0; nf < num_freq; ++nf) {
            // Compute optical depths.
            auto &sigma_HI = cross_section_HI[nf];
            auto &sigma_HeI = cross_section_HeI[nf];
            auto &sigma_HeII = cross_section_HeII[nf];

            optical_depth tau_HI{cd_in.x * sigma_HI, cd_out.x * sigma_HI};
            optical_depth tau_HeI{cd_in.y * sigma_HeI, cd_out.y * sigma_HeI};
            optical_depth tau_HeII{cd_in.z * sigma_HeII, cd_out.z * sigma_HeII};
            optical_depth tau_tot{
                tau_HI.in + tau_HeI.in + tau_HeII.in,
                tau_HI.out + tau_HeI.out + tau_HeII.out
            };

            double scaling_HI = tau_HI.cell() / tau_tot.cell();
            double scaling_HeI = tau_HeI.cell() / tau_tot.cell();
            double scaling_HeII = tau_HeII.cell() / tau_tot.cell();

            // Find the table indices for interpolation.
            auto tpos_in = log_table_index(tau_tot.in, logtau);
            auto tpos_out = log_table_index(tau_tot.out, logtau);

            auto ntau = logtau.num + 1;

            // Photo-ionization:
            auto ion_off = nf * ntau;
            auto ion = photo_table_lookup(
                tpos_in, tpos_out, tau_tot.cell(),
                {ion_tables.thin + ion_off, ion_tables.thick + ion_off}
            );
            rates += {ion * scaling_HI, ion * scaling_HeI, ion * scaling_HeII, 0.0};

            // Photo-heating:
            auto heat_off = nf * ntau * 3;
            auto heat_HI = photo_table_lookup(
                tpos_in, tpos_out, tau_tot.cell(),
                {heat_tables.thin + heat_off, heat_tables.thick + heat_off}
            );
            auto heat_HeI = photo_table_lookup(
                tpos_in, tpos_out, tau_tot.cell(),
                {heat_tables.thin + heat_off + ntau,
                 heat_tables.thick + heat_off + ntau}
            );
            auto heat_HeII = photo_table_lookup(
                tpos_in, tpos_out, tau_tot.cell(),
                {heat_tables.thin + heat_off + 2 * ntau,
                 heat_tables.thick + heat_off + 2 * ntau}
            );

            heat_HI *= scaling_HI;
            heat_HeI *= scaling_HeI;
            heat_HeII *= scaling_HeII;

            rates += compute_secondary_ionization(
                heat_factors + nf * 12, {heat_HI, heat_HeI, heat_HeII}, y1R, y2R
            );
        }  // end loop freq

        return rates;
    }

    __device__ void update_column_densities(
        element_data &__restrict__ data_HI, element_data &__restrict__ data_HeI,
        element_data &__restrict__ data_HeII, size_t cd_index,
        const double3 &coldens_in, const double3 &ndens_species, double path
    ) {
        auto &&[nHI, nHeI, nHeII] = ndens_species;
        double3 coldens_out = {
            coldens_in.x + nHI * path, coldens_in.y + nHeI * path,
            coldens_in.z + nHeII * path
        };
        data_HI.column_density[cd_index] = coldens_out.x;
        data_HeI.column_density[cd_index] = coldens_out.y;
        data_HeII.column_density[cd_index] = coldens_out.z;
    }

    // Compute the photoionization rate for a given cell based on the incoming column
    // density and the pre-computed photoionization tables.
    __device__ void update_photo_rates(
        element_data &__restrict__ data_HI, element_data &__restrict__ data_HeI,
        element_data &__restrict__ data_HeII, double *__restrict__ photo_heating,
        size_t cd_index, size_t ph_index, const double3 &coldens_in,
        const double3 &ndens_species, double xHII, double scale,
        const double *__restrict__ heat_factors,
        const photo_tables<> &__restrict__ ion_tables,
        const photo_tables<> &__restrict__ heat_tables, const linspace<> &logtau,
        size_t num_freq
    ) {
        auto &&[nHI, nHeI, nHeII] = ndens_species;
        double3 coldens_out = {
            data_HI.column_density[cd_index],
            data_HeI.column_density[cd_index],
            data_HeII.column_density[cd_index],
        };

        auto &&[y1R, y2R] = fit_factors(xHII);
        auto rates = compute_photo_rates(
            coldens_in, coldens_out, data_HI.cross_section, data_HeI.cross_section,
            data_HeII.cross_section, heat_factors, y1R, y2R, ion_tables, heat_tables,
            logtau, num_freq
        );

        // Rescale the photo rates by the flux strength normalized per volume (scale)
        // and per neutral density (part of the photon-conserving rate prescription)
        // and add it to the global array
        // FIXME: potentially a problem if the fraction value is close to zero.
        auto atomic_update = [=](double *data, double rate, double ndens) {
            assert(ndens > 0.0);
            atomicAdd(data + ph_index, rate * scale / ndens);
        };

        atomic_update(data_HI.photo_ionization, rates.ionization_HI, nHI);
        atomic_update(data_HeI.photo_ionization, rates.ionization_HeI, nHeI);
        atomic_update(data_HeII.photo_ionization, rates.ionization_HeII, nHeII);
        atomic_update(photo_heating, rates.heating, 1.0);
    }

    // Raytracing operation on a given cell, identified by (q, s). This is performed by
    // a single thread. Threads may call this function multiple times if required to
    // cover the full q-shell.
    __device__ void raytrace(
        int q, int s, int i0, int j0, int k0, double strength, double dr, double R_max,
        element_data &__restrict__ data_HI, element_data &__restrict__ data_HeI,
        element_data &__restrict__ data_HeII, double *__restrict__ photo_heating,
        const density_maps &densities, double *__restrict__ heat_factors,
        const photo_tables<> &__restrict__ ion_tables,
        const photo_tables<> &__restrict__ heat_tables, const linspace<double> &logtau,
        size_t m1, size_t num_freq
    ) {
        auto &&[di, dj, dk] = linthrd2cart(q, s);

        // Since the grid is periodic, we limit the maximum size of the raytraced
        // region to a cube as large as the mesh around the source. See line 93 of
        // evolve_source in C2Ray, this size will depend on if the mesh is even or
        // odd. Basically the idea is that you never touch a cell which is outside a
        // cube of length ~N centered on the source
        // Only do cell if it is within the (shifted under periodicity)
        // grid, i.e. at most ~N cells away from the source
        int ll = -m1 / 2;
        int lr = m1 % 2 - 1 - ll;
        if ((di < ll) || (di > lr) || (dj < ll) || (dj > lr) || (dk < ll) || (dk > lr))
            return;

#if !defined(PERIODIC)
        // When not in periodic mode, only treat cell if its in the grid
        if (!in_box(i0 + di, j0 + dj, k0 + dk, m1)) return;
#endif
        auto dist2 =
            (dr * di) * (dr * di) + (dr * dj) * (dr * dj) + (dr * dk) * (dr * dk);
        // Reducing the following calculation changes the numerical precision of
        // the result, albeit the physical result doesn't.
        if (dist2 / (dr * dr) > R_max * R_max) return;

        // FIXME: hard coded indices for cross sections
        cell_interpolator interp{di, dj, dk};
        auto cd_in_HI =
            interp.interpolate(data_HI.shared_cdens, data_HI.cross_section[0]);
        auto cd_in_HeI =
            interp.interpolate(data_HeI.shared_cdens, data_HeI.cross_section[1]);
        auto cd_in_HeII =
            interp.interpolate(data_HeII.shared_cdens, data_HeII.cross_section[27]);

        auto path = path_in_cell(di, dj, dk) * dr;
        auto vol = 4 * c::pi<> * dist2 * path;

        // Map to periodic grid
        const auto index = ravel_index(i0 + di, j0 + dj, k0 + dk, m1);
        const auto q_off = cells_to_shell(q - 1);

        // Get local number density of HI, HeI, and HeII
        auto ns = densities.get(index);

        // Update the column densities for the current cell.
        update_column_densities(
            data_HI, data_HeI, data_HeII, q_off + s, {cd_in_HI, cd_in_HeI, cd_in_HeII},
            ns, path
        );

        constexpr double max_coldens = 2e29;
        if (cd_in_HI > max_coldens || cd_in_HeI > max_coldens ||
            cd_in_HeII > max_coldens)
            return;

        // Hydrogen ionized fraction is used to compute the Ricotti form factors.
        auto xHII = densities.xHII[index];

        // Use column densities to compute the photo-ionization and -heating rates.
        update_photo_rates(
            data_HI, data_HeI, data_HeII, photo_heating, q_off + s, index,
            {cd_in_HI, cd_in_HeI, cd_in_HeII}, ns, xHII, strength / vol, heat_factors,
            ion_tables, heat_tables, logtau, num_freq
        );
    }

}  // namespace

namespace asora {
    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        size_t m1, double dr, double R_max, int q_max, size_t ns_start, size_t num_src,
        int *src_pos, double *__restrict__ src_flux, element_data data_HI,
        element_data data_HeI, element_data data_HeII,
        double *__restrict__ photo_heating, double *__restrict__ heat_factors,
        density_maps densities, photo_tables<> ion_tables, photo_tables<> heat_tables,
        linspace<double> logtau, size_t num_freq
    ) {
        /* The raytracing kernel proceeds as follows:
         * 1. Select the source based on the thread-block number
         * 2. Loop over the asora q-shells around the source, up to q_max
         * 3. For each shell, threads independently raytrace on all cells
         * 4. Before moving to the next q-shell, threads are synchronized to ensure
         * causality
         */

        // Source identifier: one source per thread-block.
        const size_t ns = ns_start + blockIdx.x;

        // Ensure the source index is valid
        if (ns >= num_src) return;

        // Get source properties.
        const auto i0 = src_pos[3 * ns + 0];
        const auto j0 = src_pos[3 * ns + 1];
        const auto k0 = src_pos[3 * ns + 2];
        const auto strength = src_flux[ns];

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block needs its own copy of the array)
        size_t cd_offset = blockIdx.x * cells_to_shell(q_max);

        data_HI.column_density += cd_offset;
        data_HeI.column_density += cd_offset;
        data_HeII.column_density += cd_offset;

        // Calculate column density and photoionization rate for the source cell.
        // This is done separately from the main loop because to take advantage of
        // some simplifications.
        if (threadIdx.x == 0) {
            const auto index = ravel_index(i0, j0, k0, m1);
            auto ns = densities.get(index);
            auto xHII = densities.xHII[index];
            update_column_densities(
                data_HI, data_HeI, data_HeII, 0, {0.0, 0.0, 0.0}, ns, 0.5 * dr
            );
            update_photo_rates(
                data_HI, data_HeI, data_HeII, photo_heating, 0, index, {0.0, 0.0, 0.0},
                ns, xHII, strength / (dr * dr * dr), heat_factors, ion_tables,
                heat_tables, logtau, num_freq
            );
        }
        __syncthreads();

        // Loop over ASORA q-shells and each thread does raytracing on one or more
        // cells. "s" is the index in the range [0, ..., 4q^2 + 1] that gets mapped
        // to the cells in the shell. The threads are usually fewer than the number
        // of cells, therefore they can do additional work. (q, s) indexing is
        // mapped to the (i, j, k) indexing of the cells via the mapping described
        // in the paper.
        for (int q = 1; q <= q_max; ++q) {
            data_HI.partition_column_density(q);
            data_HeI.partition_column_density(q);
            data_HeII.partition_column_density(q);

            int s = threadIdx.x;
            while (static_cast<size_t>(s) < cells_in_shell(q)) {
                raytrace(
                    q, s, i0, j0, k0, strength, dr, R_max, data_HI, data_HeI, data_HeII,
                    photo_heating, densities, heat_factors, ion_tables, heat_tables,
                    logtau, m1, num_freq
                );
                s += blockDim.x;
            }
            __syncthreads();
        }
    }

    // ========================================================================
    // Raytrace all sources and add up ionization rates
    // ========================================================================
    void do_all_sources_gpu(
        double R, const double *sig_HI, const double *sig_HeI, const double *sig_HeII,
        const double *heat_factors, size_t num_freq, double dr, const double *xHII_av,
        const double *xHeII_av, const double *xHeIII_av, double *phion_HI,
        double *phion_HeI, double *phion_HeII, double *pheat, size_t num_src, size_t m1,
        double minlogtau, double dlogtau, size_t num_tau, size_t grid_size,
        size_t block_size
    ) {
        device::check_initialized();

        // Size of grid data
        auto n_cells = m1 * m1 * m1;

        // Initialize and copy density data.
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::fraction_HII, xHII_av},
                 std::pair{buffer_tag::fraction_HeII, xHeII_av},
                 std::pair{buffer_tag::fraction_HeIII, xHeIII_av},
             })
            device::ensure_transfer<double>(tag, data, n_cells);

        // Initialize and set to zero photo rate data.
        for (auto tag : {
                 buffer_tag::photo_ionization_HI,
                 buffer_tag::photo_ionization_HeI,
                 buffer_tag::photo_ionization_HeII,
                 buffer_tag::photo_heating,
             }) {
            device::ensure<double>(tag, n_cells);
            auto buf = device::get(tag);
            safe_cuda(cudaMemset(buf.view<double>().data(), 0, buf.size()));
        }

        // Initialize and copy cross section data.
        // TODO: Move cross sections and form factors to device memory from outside.
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::cross_section_HI, sig_HI},
                 std::pair{buffer_tag::cross_section_HeI, sig_HeI},
                 std::pair{buffer_tag::cross_section_HeII, sig_HeII},
                 std::pair{buffer_tag::heating_form_factors, heat_factors},
             })
            device::ensure_transfer<double>(tag, data, num_freq);

        // Determine how large the octahedron should be, based on the raytracing
        // radius. Currently, this is set s.t. the radius equals the distance from
        // the source to the middle of the faces of the octahedron. To raytrace the
        // whole box, the octahedron must be 1.5*N in size
        int q_max = std::ceil(c::sqrt3<> * std::min(R, c::sqrt3<> * m1 / 2.0));

        // Allocate memory for column density calculations.
        for (auto tag : {
                 buffer_tag::column_density_HI,
                 buffer_tag::column_density_HeI,
                 buffer_tag::column_density_HeII,
             })
            device::ensure<double>(tag, grid_size * cells_to_shell(q_max));

        if (!device::contains(buffer_tag::source_flux) ||
            !device::contains(buffer_tag::source_position))
            throw std::runtime_error(
                "Source properties must be allocated on the device before calling "
                "do_all_sources_gpu"
            );
        auto src_flux_d = get_data_view<double>(buffer_tag::source_flux);
        auto src_pos_d = get_data_view<int>(buffer_tag::source_position);

        auto densities = make_density_maps();

        auto data_HI = make_element_data(
            buffer_tag::photo_ionization_HI, buffer_tag::column_density_HI,
            buffer_tag::cross_section_HI
        );
        auto data_HeI = make_element_data(
            buffer_tag::photo_ionization_HeI, buffer_tag::column_density_HeI,
            buffer_tag::cross_section_HeI
        );
        auto data_HeII = make_element_data(
            buffer_tag::photo_ionization_HeII, buffer_tag::column_density_HeII,
            buffer_tag::cross_section_HeII
        );
        auto pheat_d = get_data_view<double>(buffer_tag::photo_heating);
        auto heat_facts_d = get_data_view<double>(buffer_tag::heating_form_factors);

        photo_tables ion_tables{
            get_data_view<double>(buffer_tag::photo_ion_thin_table),
            get_data_view<double>(buffer_tag::photo_ion_thick_table)
        };
        photo_tables heat_tables{
            get_data_view<double>(buffer_tag::photo_heat_thin_table),
            get_data_view<double>(buffer_tag::photo_heat_thick_table)
        };

        linspace<double> logtau{minlogtau, dlogtau, static_cast<size_t>(num_tau)};

        // Loop over batches of sources
        for (size_t ns = 0; ns < num_src; ns += grid_size) {
            // Raytrace the current batch of sources in parallel
            // Consecutive kernel launches are in the same stream and so are
            // serialized
            evolve0D_gpu<<<grid_size, block_size>>>(
                m1, dr, R, q_max, ns, num_src, src_pos_d, src_flux_d, data_HI, data_HeI,
                data_HeII, pheat_d, heat_facts_d, densities, ion_tables, heat_tables,
                logtau, num_freq
            );

            safe_cuda(cudaPeekAtLastError());
        }

        // Copy the accumulated ionization rates back to the host
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::photo_ionization_HI, phion_HI},
                 std::pair{buffer_tag::photo_ionization_HeI, phion_HeI},
                 std::pair{buffer_tag::photo_ionization_HeII, phion_HeII},
                 std::pair{buffer_tag::photo_heating, pheat},
             }) {
            auto buf = device::get(tag);
            buf.copyToHost(data);
        }
    }

}  // namespace asora
