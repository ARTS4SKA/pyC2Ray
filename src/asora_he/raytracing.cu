#include "raytracing.cuh"

#include "../asora/memory.h"
#include "../asora/utils.cuh"

#include <cuda_runtime.h>

#include <cuda/std/span>
#include <exception>

namespace asora {

    __device__ void element_data::partition_column_density(int q) {
        shared_cdens = {
            column_density + asora::cells_to_shell(q - 2),
            column_density + asora::cells_to_shell(q - 3),
            column_density + asora::cells_to_shell(q - 4)
        };
    }

    __device__ cuda::std::array<double, 3> density_maps::get(size_t index) const {
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

    template <typename T>
    T *get_data_view(asora::buffer_tag tag) {
        return asora::device::get(tag).view<T>().data();
    }

    element_data make_element_data(
        asora::buffer_tag ion, asora::buffer_tag heat, asora::buffer_tag cdens,
        asora::buffer_tag sigma, size_t first_bin
    ) {
        return {
            get_data_view<double>(ion),
            get_data_view<double>(heat),
            get_data_view<double>(cdens),
            get_data_view<double>(sigma),
            first_bin,
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

    __device__ void update_photo_rates(
        element_data &__restrict__ data_HI, element_data &__restrict__ data_HeI,
        element_data &__restrict__ data_HeII, size_t cd_index, size_t ph_index,
        const cuda::std::array<double, 3> &coldens_in,  // for HI, HeI, HeII
        const cuda::std::array<double, 3> &ndens_in,    // for HI, HeI, HeII
        double path, double strength, double vol,
        const photo_tables &__restrict__ ion_tables,
        const photo_tables &__restrict__ heat_tables, const linspace<double> &logtau,
        size_t num_freq
    ) {
        auto &&[cd_in_HI, cd_in_HeI, cd_in_HeII] = coldens_in;
        auto &&[nHI, nHeI, nHeII] = ndens_in;

        auto &cd_out_HI = data_HI.column_density[cd_index];
        auto &cd_out_HeI = data_HeI.column_density[cd_index];
        auto &cd_out_HeII = data_HeII.column_density[cd_index];

        cd_out_HI = cd_in_HI + nHI * path;
        cd_out_HeI = cd_in_HeI + nHeI * path;
        cd_out_HeII = cd_in_HeII + nHeII * path;

        // Default-initialized to {0, 0}
        using photo_rate = cuda::std::pair<double, double>;
        photo_rate rate_HI;
        photo_rate rate_HeI;
        photo_rate rate_HeII;

        using tau = cuda::std::pair<double, double>;
        auto get_tau = [](size_t nf, const element_data &data, double cd_in,
                          double cd_out) -> tau {
            if (nf < data.first_bin) return {0.0, 0.0};
            auto sigma = data.cross_section[nf];
            return {cd_in * sigma, cd_out * sigma};
        };

        // frequency loop
        for (size_t nf = 0; nf < num_freq; ++nf) {
            // Compute optical depths
            auto tau_HI = get_tau(nf, data_HI, cd_in_HI, cd_out_HI);
            auto tau_HeI = get_tau(nf, data_HeI, cd_in_HeI, cd_out_HeI);
            auto tau_HeII = get_tau(nf, data_HeII, cd_in_HeII, cd_out_HeII);

            tau tau_tot{
                tau_HI.first + tau_HeI.first + tau_HeII.first,
                tau_HI.second + tau_HeI.second + tau_HeII.second
            };

            // TODO: potentially a problem if the fraction value is close to
            // zero.
            auto norm = strength / vol / (tau_tot.second - tau_tot.first);
            auto mul_HI = (tau_HI.second - tau_HI.first) * norm;
            auto mul_HeI = (tau_HeI.second - tau_HeI.first) * norm;
            auto mul_HeII = (tau_HeII.second - tau_HeII.first) * norm;

            auto nf_offset = nf * (logtau.num + 1);
            auto phi = asora::photo_rates_gpu(
                tau_tot.first, tau_tot.second,
                {ion_tables.thin + nf_offset, ion_tables.thick + nf_offset}, logtau
            );
            // TODO: heating requires more tables than just one because of the
            // frequency dependency. Probably solution is to move this inside
            // the if else condition (look at the radiation_tables.f90 line 322)
            auto heat = asora::photo_rates_gpu(
                tau_tot.first, tau_tot.second,
                {heat_tables.thin + nf_offset, heat_tables.thick + nf_offset}, logtau
            );

            // Assign the photo-ionization and heating rates to each element
            // (part of the photon-conserving rate prescription)
            rate_HI.first += phi * mul_HI;
            rate_HeI.first += phi * mul_HeI;
            rate_HeII.first += phi * mul_HeII;
            rate_HI.second += heat * mul_HI;
            rate_HeI.second += heat * mul_HeI;
            rate_HeII.second += heat * mul_HeII;
        }  // end loop freq

        // Add the computed ionization and heating rate to the array atomically
        atomicAdd(data_HI.photo_ionization + ph_index, rate_HI.first / nHI);
        atomicAdd(data_HeI.photo_ionization + ph_index, rate_HeI.first / nHeI);
        atomicAdd(data_HeII.photo_ionization + ph_index, rate_HeII.first / nHeII);
        atomicAdd(data_HI.photo_heating + ph_index, rate_HI.second / nHI);
        atomicAdd(data_HeI.photo_heating + ph_index, rate_HeI.second / nHeI);
        atomicAdd(data_HeII.photo_heating + ph_index, rate_HeII.second / nHeII);
    }

    __device__ void raytrace(
        int q, int s, int i0, int j0, int k0, double strength, double dr, double R_max,
        element_data &__restrict__ data_HI, element_data &__restrict__ data_HeI,
        element_data &__restrict__ data_HeII, const density_maps &densities,
        const photo_tables &__restrict__ ion_tables,
        const photo_tables &__restrict__ heat_tables, const linspace<double> &logtau,
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

        // Split column density in memory banks corresponding to shells q-1, q-2, q-3.
        // FIXME: This is the same for each s....

        // Column density of HI, HeI and HeII to the cell
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

        auto dist2 =
            (dr * di) * (dr * di) + (dr * dj) * (dr * dj) + (dr * dk) * (dr * dk);
        if (dist2 / (dr * dr) > R_max * R_max) return;

        // Map to periodic grid
        const auto index = mem_offset(i0 + di, j0 + dj, k0 + dk, m1);
        const auto q_off = cells_to_shell(q - 1);

        // Get local number density of HI, HeI, and HeII
        auto ns = densities.get(index);

        auto path = path_in_cell(di, dj, dk) * dr;
        auto vol = 4 * c::pi<> * dist2 * path;

        update_photo_rates(
            data_HI, data_HeI, data_HeII, q_off + s, index,
            {cd_in_HI, cd_in_HeI, cd_in_HeII}, ns, path, strength, vol, ion_tables,
            heat_tables, logtau, num_freq
        );
    }

}  // namespace

namespace asora {
    // ========================================================================
    // Raytrace all sources and add up ionization rates
    // ========================================================================
    void do_all_sources_gpu(
        double R, const double *sig_HI, const double *sig_HeI, const double *sig_HeII,
        size_t num_bin_1, size_t num_bin_2, size_t num_freq, double dr,
        const double *xHII_av, const double *xHeII_av, const double *xHeIII_av,
        double *phi_ion_HI, double *phi_ion_HeI, double *phi_ion_HeII,
        double *phi_heat_HI, double *phi_heat_HeI, double *phi_heat_HeII,
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
             }) {
            if (!device::contains(tag)) device::add<double>(tag, n_cells);
            auto buf = device::get(tag);
            buf.copyFromHost(data, buf.size());
        }

        // Initialize and set to zero photo rate data.
        for (auto tag : {
                 buffer_tag::photo_ionization_HI,
                 buffer_tag::photo_ionization_HeI,
                 buffer_tag::photo_ionization_HeII,
                 buffer_tag::photo_heating_HI,
                 buffer_tag::photo_heating_HeI,
                 buffer_tag::photo_heating_HeII,
             }) {
            if (!device::contains(tag)) device::add<double>(tag, n_cells);
            auto buf = device::get(tag);
            safe_cuda(cudaMemset(buf.view<double>().data(), 0, buf.size()));
        }

        // Initialize and copy cross section data.
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::cross_section_HI, sig_HI},
                 std::pair{buffer_tag::cross_section_HeI, sig_HeI},
                 std::pair{buffer_tag::cross_section_HeII, sig_HeII},
             }) {
            if (!device::contains(tag)) device::add<double>(tag, num_freq);
            auto buf = device::get(tag);
            buf.copyFromHost(data, buf.size());
        }

        // Determine how large the octahedron should be, based on the raytracing
        // radius. Currently, this is set s.t. the radius equals the distance from
        // the source to the middle of the faces of the octahedron. To raytrace the
        // whole box, the octahedron must be 1.5*N in size
        int q_max = std::ceil(c::sqrt3<> * min(R, c::sqrt3<> * m1 / 2.0));

        // Allocate memory for column density calculations.
        for (auto tag : {
                 buffer_tag::column_density_HI,
                 buffer_tag::column_density_HeI,
                 buffer_tag::column_density_HeII,
             }) {
            if (!device::contains(tag))
                device::add<double>(tag, grid_size * cells_to_shell(q_max));
        }

        auto src_flux_d = get_data_view<double>(buffer_tag::source_flux);
        auto src_pos_d = get_data_view<int>(buffer_tag::source_position);

        auto densities = make_density_maps();

        auto data_HI = make_element_data(
            buffer_tag::photo_ionization_HI, buffer_tag::photo_heating_HI,
            buffer_tag::column_density_HI, buffer_tag::cross_section_HI, 0
        );
        auto data_HeI = make_element_data(
            buffer_tag::photo_ionization_HeI, buffer_tag::photo_heating_HeI,
            buffer_tag::column_density_HeI, buffer_tag::cross_section_HeI, num_bin_1
        );
        auto data_HeII = make_element_data(
            buffer_tag::photo_ionization_HeII, buffer_tag::photo_heating_HeII,
            buffer_tag::column_density_HeII, buffer_tag::cross_section_HeII,
            num_bin_1 + num_bin_2
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
                 std::pair{buffer_tag::photo_ionization_HI, phi_ion_HI},
                 std::pair{buffer_tag::photo_ionization_HeI, phi_ion_HeI},
                 std::pair{buffer_tag::photo_ionization_HeII, phi_ion_HeII},
                 std::pair{buffer_tag::photo_heating_HI, phi_heat_HI},
                 std::pair{buffer_tag::photo_heating_HeI, phi_heat_HeI},
                 std::pair{buffer_tag::photo_heating_HeII, phi_heat_HeII},
             }) {
            auto buf = device::get(tag);
            buf.copyToHost(data, buf.size());
        }
    }

    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        size_t m1, double dr, double R_max, int q_max, size_t ns_start, size_t num_src,
        int *src_pos, double *src_flux, element_data data_HI, element_data data_HeI,
        element_data data_HeII, density_maps densities, photo_tables ion_tables,
        photo_tables heat_tables, linspace<double> logtau, size_t num_freq
    ) {
        /* The raytracing kernel proceeds as follows:
        1. Select the source based on the block number (within the batch = the grid)
        2. Loop over the asora q-cells around the source, up to q_max (loop "A")
        3. Inside each shell, threads independently do all cells, possibly requiring
        multiple iterations if the block size is smaller than the number of cells in
        the shell (loop "B")
        4. After each shell, the threads are synchronized to ensure that causality
        is respected
        */

        // Source number = Start of batch + block number (each block does one
        // source)
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

        if (threadIdx.x == 0) {
            const auto index = mem_offset(i0, j0, k0, m1);
            auto ns = densities.get(index);
            update_photo_rates(
                data_HI, data_HeI, data_HeII, 0, index, {0.0, 0.0, 0.0}, ns, 0.5 * dr,
                strength, dr * dr * dr, ion_tables, heat_tables, logtau, num_freq
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
            while (s < cells_in_shell(q)) {
                raytrace(
                    q, s, i0, j0, k0, strength, dr, R_max, data_HI, data_HeI, data_HeII,
                    densities, ion_tables, heat_tables, logtau, m1, num_freq
                );
                s += blockDim.x;
            }
            __syncthreads();
        }
    }

}  // namespace asora
