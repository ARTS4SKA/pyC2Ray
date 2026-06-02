! This module contains routines having to do with the calculation of the thermal evolution of a single point/cell.

module thermalevolution

   !use precision, only: real64
   use, intrinsic :: iso_fortran_env, only: real64
   use chem_utils, only: get_energy, get_temperature, cooling_rate, cosmo_cool

   implicit none

   !> Thermal: minimum temperature [K]
   real(kind=real64), parameter :: minitemp = 1.0_real64
   !> Thermal: fraction of the cooling time step below which no iteration is done
   real(kind=real64), parameter :: relative_denergy = 0.1_real64
   !> adiabatic index
   real(kind=real64), public, parameter :: gamma = 5.0_real64/3.0_real64

contains
   ! calculates the thermal evolution of one grid point
   subroutine thermal(dt, end_temper, avg_temper, ndens_el, ndens_atom, heat, Hz, cosmo_only)

      ! The time step
      real(kind=real64), intent(in) :: dt
      ! end time temperature of the cell
      real(kind=real64), intent(inout) :: end_temper
      ! average temperature of the cell
      real(kind=real64), intent(inout) :: avg_temper
      ! Electron density of the cell
      real(kind=real64), intent(in) :: ndens_el
      ! Number density of atoms of the cell
      real(kind=real64), intent(in) :: ndens_atom
      ! Heating rate
      real(kind=real64), intent(in) :: heat
      ! Hubble function at the corresponding redshift (in cgs)
      real(kind=real64), intent(in) :: Hz
      ! Whether to only include cosmological cooling
      logical(kind=4), intent(in) :: cosmo_only

      ! initial temperature
      real(kind=real64) :: initial_temp
      ! timestep taken to solve the ODE
      real(kind=real64) :: dt_ODE
      ! timestep related to thermal timescale
      real(kind=real64) :: dt_thermal
      ! record the time elapsed
      real(kind=real64) :: cumulative_time
      ! internal energy of the cell
      real(kind=real64) :: internal_energy
      ! thermal timescale, used to calculate the thermal timestep
      real(kind=real64) :: thermal_timescale
      ! heating rate
      real(kind=real64) :: heating
      ! cooling rate
      real(kind=real64) :: cooling
      ! difference of heating and cooling rate
      real(kind=real64) :: thermal_rate
      ! total rate of energy change
      real(kind=real64) :: rate
      ! Counter of number of thermal timesteps taken
      integer :: niter

      ! Thermal process is only done if the temperature of the cell is larger than the minimum temperature requirement
      if (end_temper <= minitemp) then
         avg_temper = end_temper
         return
      end if

      ! heating rate
      heating = heat

      ! Find initial internal energy
      internal_energy = get_energy(end_temper, ndens_atom, ndens_el, gamma)

      ! stores the time elapsed is done
      cumulative_time = 0.0

      ! initialize the counter
      niter = 0

      ! thermal process loop begins
      do while (niter < 10000 .and. cumulative_time < dt*(1.0 - 1e-6))

         ! update cooling rate from cooling tables and add adeabatic cooling (cosmological expansion)
         ! TODO: check that cosmo_cool_rate is not to be updated at each step
         cooling = cosmo_cool(internal_energy, Hz)
         if (.not. cosmo_only) then
            cooling = cooling + cooling_rate(ndens_atom, ndens_el, end_temper)
         end if
         rate = heating - cooling

         ! Find total energy change rate
         thermal_rate = max(1d-50, abs(rate))

         ! Calculate time step needed to limit energy change to a fraction relative_denergy
         dt_thermal = relative_denergy*internal_energy/thermal_rate

         ! Time step to large, change it to dt_thermal. Make sure we do not integrate for longer than the total time step
         dt_ODE = min(dt_thermal, dt - cumulative_time)

         ! Find new internal energy density
         internal_energy = internal_energy + dt_ODE*rate

         ! Update avg_temper sum (first part of dt_thermal sub time step)
         avg_temper = avg_temper + 0.5*end_temper*dt_ODE

         ! Find new temperature from the internal energy density
         end_temper = get_temperature(internal_energy, ndens_atom, ndens_el, gamma)

         ! Update avg_temper sum (second part of dt_thermal sub time step)
         avg_temper = avg_temper + 0.5*end_temper*dt_ODE

         ! Update fractional cumulative_time
         cumulative_time = cumulative_time + dt_ODE
         niter = niter + 1

         ! Take measures if temperature drops below minitemp
         if (end_temper < minitemp) then
            internal_energy = get_energy(minitemp, ndens_atom, ndens_el, gamma)
            end_temper = minitemp
            exit
         end if

      end do

      ! Calculate the final temperature
      if (cumulative_time > 0) then
         avg_temper = avg_temper/cumulative_time
      end if

   end subroutine thermal

end module thermalevolution
