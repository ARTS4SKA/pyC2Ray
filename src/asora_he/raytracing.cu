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
        // TODO: need to expose this to parameters.yml
        constexpr double abu_he_mass = 0.2486;

        auto np = ndens[index];
        auto nHI = np * (1.0 - abu_he_mass) * (1.0 - xHII[index]);
        auto nHeI = np * abu_he_mass * (1.0 - xHeII[index] - xHeIII[index]);
        auto nHeII = np * abu_he_mass * xHeII[index];

        return {nHI, nHeI, nHeII};
    }

}  // namespace asora

namespace {

    using namespace asora;

    using cross_section_histogram = cuda::std::span<double>;

    element_data make_element_data(
        asora::buffer_tag ion, asora::buffer_tag heat, asora::buffer_tag cdens,
        asora::buffer_tag sigma
    ) {
        return {
            get_data_view<double>(ion),
            get_data_view<double>(heat),
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

    struct photo_rate {
        double ionization = 0.0;
        double heating = 0.0;

        /// Add the contribution from a cell to the total photoionization and heating
        /// rates.
        __device__ void add(
            double ion, double heat, const optical_depth &tau,
            const optical_depth &tau_tot
        ) {
            assert(tau_tot.cell() > 0.0);
            // FIXME: potentially a problem if tau_tot.cell is close to zero.
            auto scale = tau.cell() / tau_tot.cell();
            ionization += ion * scale;
            heating += heat * scale;
        }
    };

    // Compute the photoionization and heating rates for a given cell based on the
    // incoming and outgoing column densities, cross sections, and pre-computed
    // photoionization tables.
    __device__ cuda::std::array<photo_rate, 3> compute_photo_rates(
        const double3 &cd_in, const double3 &cd_out,
        const double *__restrict__ cross_section_HI,
        const double *__restrict__ cross_section_HeI,
        const double *__restrict__ cross_section_HeII,
        const photo_tables<> &__restrict__ ion_tables,
        const photo_tables<> &__restrict__ heat_tables, const linspace<> &logtau,
        size_t num_freq
    ) {
        photo_rate rate_HI;
        photo_rate rate_HeI;
        photo_rate rate_HeII;

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

            // Find the table indices for interpolation.
            auto tpos_in = log_table_index(tau_tot.in, logtau);
            auto tpos_out = log_table_index(tau_tot.out, logtau);

            auto nf_offset = nf * (logtau.num + 1);
            auto phi = photo_table_lookup(
                tpos_in, tpos_out, tau_tot.cell(),
                {ion_tables.thin + nf_offset, ion_tables.thick + nf_offset}
            );
            auto heat = photo_table_lookup(
                tpos_in, tpos_out, tau_tot.cell(),
                {heat_tables.thin + nf_offset, heat_tables.thick + nf_offset}
            );

            // Update the photoionization and heating rates for each species.
            rate_HI.add(phi, heat, tau_HI, tau_tot);
            rate_HeI.add(phi, heat, tau_HeI, tau_tot);
            rate_HeII.add(phi, heat, tau_HeII, tau_tot);
        }  // end loop freq

        return {std::move(rate_HI), std::move(rate_HeI), std::move(rate_HeII)};
    }

    // Compute the photoionization rate for a given cell based on the incoming column
    // density and the pre-computed photoionization tables.
    __device__ void update_photo_rates(
        element_data &__restrict__ data_HI, element_data &__restrict__ data_HeI,
        element_data &__restrict__ data_HeII, size_t cd_index, size_t ph_index,
        const double3 &coldens_in, const double3 &ndens_in, double path, double scale,
        const photo_tables<> &__restrict__ ion_tables,
        const photo_tables<> &__restrict__ heat_tables, const linspace<> &logtau,
        size_t num_freq
    ) {
        auto &&[nHI, nHeI, nHeII] = ndens_in;
        double3 coldens_out = {
            coldens_in.x + nHI * path, coldens_in.y + nHeI * path,
            coldens_in.z + nHeII * path
        };
        data_HI.column_density[cd_index] = coldens_out.x;
        data_HeI.column_density[cd_index] = coldens_out.y;
        data_HeII.column_density[cd_index] = coldens_out.z;

        auto &&[rate_HI, rate_HeI, rate_HeII] = compute_photo_rates(
            coldens_in, coldens_out, data_HI.cross_section, data_HeI.cross_section,
            data_HeII.cross_section, ion_tables, heat_tables, logtau, num_freq
        );

        // Rescale the photo rates by the flux strength normalized per volume (scale)
        // and per neutral density (part of the photon-conserving rate prescription)
        // and add it to the global array
        // FIXME: potentially a problem if the fraction value is close to zero.
        auto atomic_update = [=](double *data, double rate, double ndens) {
            assert(ndens > 0.0);
            atomicAdd(data + ph_index, rate * scale / ndens);
        };

        atomic_update(data_HI.photo_ionization, rate_HI.ionization, nHI);
        atomic_update(data_HI.photo_heating, rate_HI.heating, nHI);

        atomic_update(data_HeI.photo_ionization, rate_HeI.ionization, nHeI);
        atomic_update(data_HeI.photo_heating, rate_HeI.heating, nHeI);

        atomic_update(data_HeII.photo_ionization, rate_HeII.ionization, nHeII);
        atomic_update(data_HeII.photo_heating, rate_HeII.heating, nHeII);
    }

    // Raytracing operation on a given cell, identified by (q, s). This is performed by
    // a single thread. Threads may call this function multiple times if required to
    // cover the full q-shell.
    __device__ void raytrace(
        int q, int s, int i0, int j0, int k0, double strength, double dr, double R_max,
        element_data &__restrict__ data_HI, element_data &__restrict__ data_HeI,
        element_data &__restrict__ data_HeII, const density_maps &densities,
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

        cell_interpolator interp{di, dj, dk};
        auto cd_in_HI =
            interp.interpolate(data_HI.shared_cdens, data_HI.cross_section[0]);
        auto cd_in_HeI =
            interp.interpolate(data_HeI.shared_cdens, data_HeI.cross_section[0]);
        auto cd_in_HeII =
            interp.interpolate(data_HeII.shared_cdens, data_HeII.cross_section[0]);

        // Compute photoionization rates from column density.
        // WARNING: for now this is limited to the grey-opacity
        // test case source
        constexpr double max_coldens = 2e30;
        if (cd_in_HI > max_coldens || cd_in_HeI > max_coldens ||
            cd_in_HeII > max_coldens)
            return;

        auto path = path_in_cell(di, dj, dk) * dr;
        auto vol = 4 * c::pi<> * dist2 * path;

        // Map to periodic grid
        const auto index = ravel_index(i0 + di, j0 + dj, k0 + dk, m1);
        const auto q_off = cells_to_shell(q - 1);

        // Get local number density of HI, HeI, and HeII
        auto ns = densities.get(index);

        update_photo_rates(
            data_HI, data_HeI, data_HeII, q_off + s, index,
            {cd_in_HI, cd_in_HeI, cd_in_HeII}, ns, path, strength / vol, ion_tables,
            heat_tables, logtau, num_freq
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
        int *src_pos, double *src_flux, element_data data_HI, element_data data_HeI,
        element_data data_HeII, density_maps densities, photo_tables<> ion_tables,
        photo_tables<> heat_tables, linspace<double> logtau, size_t num_freq
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
            update_photo_rates(
                data_HI, data_HeI, data_HeII, 0, index, {0.0, 0.0, 0.0}, ns, 0.5 * dr,
                strength / (dr * dr * dr), ion_tables, heat_tables, logtau, num_freq
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
                    densities, ion_tables, heat_tables, logtau, m1, num_freq
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
        size_t num_freq, double dr, const double *xHII_av, const double *xHeII_av,
        const double *xHeIII_av, double *phion_HI, double *phion_HeI,
        double *phion_HeII, double *pheat_HI, double *pheat_HeI, double *pheat_HeII,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size
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
                 buffer_tag::photo_heating_HI,
                 buffer_tag::photo_heating_HeI,
                 buffer_tag::photo_heating_HeII,
             }) {
            device::ensure<double>(tag, n_cells);
            auto buf = device::get(tag);
            safe_cuda(cudaMemset(buf.view<double>().data(), 0, buf.size()));
        }

        // Initialize and copy cross section data.
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::cross_section_HI, sig_HI},
                 std::pair{buffer_tag::cross_section_HeI, sig_HeI},
                 std::pair{buffer_tag::cross_section_HeII, sig_HeII},
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
            buffer_tag::photo_ionization_HI, buffer_tag::photo_heating_HI,
            buffer_tag::column_density_HI, buffer_tag::cross_section_HI
        );
        auto data_HeI = make_element_data(
            buffer_tag::photo_ionization_HeI, buffer_tag::photo_heating_HeI,
            buffer_tag::column_density_HeI, buffer_tag::cross_section_HeI
        );
        auto data_HeII = make_element_data(
            buffer_tag::photo_ionization_HeII, buffer_tag::photo_heating_HeII,
            buffer_tag::column_density_HeII, buffer_tag::cross_section_HeII
        );

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
                data_HeII, densities, ion_tables, heat_tables, logtau, num_freq
            );

            safe_cuda(cudaPeekAtLastError());
        }

        // Copy the accumulated ionization rates back to the host
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::photo_ionization_HI, phion_HI},
                 std::pair{buffer_tag::photo_ionization_HeI, phion_HeI},
                 std::pair{buffer_tag::photo_ionization_HeII, phion_HeII},
                 std::pair{buffer_tag::photo_heating_HI, pheat_HI},
                 std::pair{buffer_tag::photo_heating_HeI, pheat_HeI},
                 std::pair{buffer_tag::photo_heating_HeII, pheat_HeII},
             }) {
            auto buf = device::get(tag);
            buf.copyToHost(data);
        }
    }

}  // namespace asora
