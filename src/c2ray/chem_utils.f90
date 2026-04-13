! This module contains utilities routines for the calculation of the thermal and chemistry evolution.

module chem_utils
   use, intrinsic :: iso_fortran_env

   implicit none

   ! Boltzmann constant
   real(kind=real64), parameter :: k_B = 1.380649d-16     ! in erg/K units (value from astropy==6.1.7)
   ! minimum temperature [K]
   real(kind=real64), parameter :: minitemp = 1.0d0

contains

   ! Calculate the electron density
   elemental function electrondens(ndens, xhii, xheii, xheiii, abu_h, abu_he, abu_c)
      real(kind=8), intent(in) :: ndens           ! gas number density
      real(kind=8), intent(in) :: xhii             ! HI ionization fractions
      real(kind=8), intent(in) :: xheii           ! HeI ionization fractions
      real(kind=8), intent(in) :: xheiii          ! HeII ionization fractions
      real(kind=8), intent(in) :: abu_h           ! Hydrogen abundance
      ! TODO: technically abu_h = 1 - abu_he. So maybe just one term can be passed
      real(kind=8), intent(in) :: abu_he          ! Helium abundance
      real(kind=8), intent(in) :: abu_c           ! Carbon abundance

      real(kind=8) :: electrondens               ! electron number density

      electrondens = ndens*(abu_h*xhii + abu_he*(xheii + 2.0d0*xheiii) + abu_c)

   end function electrondens

   ! Calculate pressure from temperature
   elemental function get_temperature(energy, ndens, eldens, gamma) result(temp)
      real(kind=8), intent(in) :: energy    ! internal energy density
      real(kind=8), intent(in) :: ndens     ! gas number density (n_H + n_He)
      real(kind=8), intent(in) :: eldens    ! electron density
      real(kind=8), intent(in) :: gamma     ! gas constant

      real(kind=8) :: temp   ! electron number density

      temp = (gamma - 1.0d0)*energy/(k_B*(ndens + eldens))

   end function get_temperature

   ! Calculate internal energy density from temperature
   elemental function get_energy(temp, ndens, eldens, gamma) result(internal_energy)
      real(kind=8), intent(in) :: temp    ! temperature
      real(kind=8), intent(in) :: ndens   ! gas number density (n_H + n_He)
      real(kind=8), intent(in) :: eldens  ! electron density
      real(kind=8), intent(in) :: gamma   ! gas constant

      real(kind=8) :: internal_energy   ! internal energy density

      internal_energy = k_B*temp*(ndens + eldens)/(gamma - 1.0d0)

   end function get_energy

   ! TODO: this function read some tables. Need to change
   !> Calculate the cooling rate
   elemental function cooling_rate(nucldens, eldens, temp0) result(coolin)
      real(kind=8), intent(in) :: nucldens !< number density
      real(kind=8), intent(in) :: eldens !< electron density
      real(kind=8), intent(in) :: temp0 !< temperature
      real(kind=8) :: coolin

      ! TODO: need to write this, read the tables, etc...
      !real(kind=real64) :: tpos, dtpos
      !integer :: itpos,itpos1

      !tpos=(log10(temp0)-mintemp)/dtemp+1.0d0
      !itpos=min(temppoints-1,max(1,int(tpos)))
      !dtpos=tpos-real(itpos)
      !itpos1=min(temppoints,itpos+1)

      ! Cooling curve
      coolin = nucldens*eldens*temp0 !*(cie_cool(itpos)+(cie_cool(itpos1)-cie_cool(itpos))*dtpos)

   end function cooling_rate

   ! TODO: this function below could be a variable in python passed to the thermal_evolve subroutine. Using astropy for the dz/dt will also assure consistency with the cosmology in the C2Ray class.
   !> Calculates the cosmological adiabatic cooling
   elemental function cosmo_cool(e_int, Hz)
      real(kind=8), intent(in) :: e_int
      real(kind=8), intent(in) :: Hz

      real(kind=8) :: cosmo_cool

      !Cooling rate
      cosmo_cool = 2.0d0*e_int*Hz

   end function cosmo_cool

end module chem_utils
