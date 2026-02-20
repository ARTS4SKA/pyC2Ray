! This module contains routines having to do with the calculation of the thermal evolution of a single point/cell. 

module thermalevolution

  !use precision, only: real64
  use, intrinsic :: iso_fortran_env, only: real64
  use chem_utils, only: get_energy, get_temperature, cooling_rate, cosmo_cool

  implicit none

  !> Thermal: minimum temperature [K]
  real(kind=real64),parameter :: minitemp=1.0_real64
  !> Thermal: fraction of the cooling time step below which no iteration is done
  real(kind=real64),parameter :: relative_denergy=0.1_real64
  !> adiabatic index
  real(kind=real64),public,parameter :: gamma = 5.0_real64/3.0_real64


contains
  ! calculates the thermal evolution of one grid point
  subroutine thermal(dt, end_temper, avg_temper, ndens_el, ndens_atom, heat, Hz)

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
    ! cosmological cooling rate
    real(kind=real64) :: cosmo_cool_rate
    ! Counter of number of thermal timesteps taken
    integer :: i_heating
    
    ! TODO: this is for debug (consider to pass it as a variable)
    logical(kind=4) :: cosmological = .true.
    
    ! heating rate
    heating = heat

    ! Find initial internal energy
    internal_energy = get_energy(end_temper, ndens_atom, ndens_el, gamma)
    
    ! TODO: the variable cosmo_cool_rate can 
    ! Set the cosmological cooling rate
    if (cosmological) then
       cosmo_cool_rate=cosmo_cool(internal_energy, Hz)
    else
       ! Disabled for testing (non-cosmological)
       cosmo_cool_rate=0.0_real64
    endif

    ! Thermal process is only done if the temperature of the cell is larger than the minimum temperature requirement
    if (end_temper > minitemp) then

       ! stores the time elapsed is done
       cumulative_time = 0.0 
   
       ! initialize the counter
       i_heating = 0

       ! initialize time averaged temperature
       avg_temper = 0.0 

       ! initial temperature
       initial_temp = end_temper  

       ! thermal process loop begins
       do
          ! update counter              
          i_heating = i_heating+1 
         
          ! update cooling rate from cooling tables and add adeabatic cooling (cosmological expansion)
          cooling = cooling_rate(ndens_atom, ndens_el, end_temper) + cosmo_cool_rate

          ! Find total energy change rate
          thermal_rate = max(1d-50, abs(cooling-heating))

          ! Calculate thermal time scale
          thermal_timescale = internal_energy/abs(thermal_rate)

          ! Calculate time step needed to limit energy change to a fraction relative_denergy
          dt_thermal = relative_denergy*thermal_timescale

          ! Time step to large, change it to dt_thermal. Make sure we do not integrate for longer than the total time step
          dt_ODE = min(dt_thermal, dt-cumulative_time)

          ! Find new internal energy density
          internal_energy = internal_energy + dt_ODE*(heating-cooling)

          ! Update avg_temper sum (first part of dt_thermal sub time step)
          avg_temper = avg_temper + 0.5*end_temper*dt_ODE

          ! Find new temperature from the internal energy density
          end_temper = get_temperature(internal_energy, ndens_atom, ndens_el, gamma)

          ! Take measures if temperature drops below minitemp
          if (end_temper < minitemp) then
             internal_energy = get_energy(minitemp, ndens_atom, ndens_el, gamma)
             end_temper = minitemp
          endif

          ! Update avg_temper sum (second part of dt_thermal sub time step)
          avg_temper = avg_temper + 0.5*end_temper*dt_ODE
                    
          ! Update fractional cumulative_time
          cumulative_time = cumulative_time+dt_ODE
  
          ! Exit if we reach dt
          if (cumulative_time >= dt.or.abs(cumulative_time-dt) < 1e-6*dt) exit

          ! In case we spend too much time here, we exit
          if (i_heating > 10000) exit
       
       enddo
              
       ! Calculate the averaged temperature
       if (dt > 0.0) then
          avg_temper = avg_temper/dt
       else
          avg_temper = initial_temp
       endif
       
       ! Calculate the final temperature 
       end_temper = get_temperature(internal_energy, ndens_atom, ndens_el, gamma)
       
    endif
    
  end subroutine thermal
  
end module thermalevolution
