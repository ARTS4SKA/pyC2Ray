module chemistry_he
    !! Module to compute the time-averaged ionization rates and update electron density

   use, intrinsic :: iso_fortran_env, only: real64
   use thermalevolution, only: thermal_impl => thermal
   use chem_utils, only: electrondens

   use, intrinsic :: iso_c_binding, only: c_double

   implicit none

   interface
      real(c_double) function expm1(x) bind(c, name='expm1')
         import c_double
         real(c_double), intent(in), value :: x
      end function expm1
   end interface

   public :: global_pass, thermal

   real(kind=real64), parameter :: epsilon = 1d-14                    ! Double precision very small number
   real(kind=real64), parameter :: minimum_fractional_change = 1.0d-3      ! Should be a global parameter. TODO
   real(kind=real64), parameter :: minimum_fraction_of_atoms = 1.0d-8
   real(kind=real64), parameter :: minitemp = 1.0_real64                   ! minimum temperature

   ! TODO: the variables here below need to be inported by the module rather then being hard-coded
   ! cross section constants
   real(kind=real64), parameter :: sigma_H_at_HeI = 1.238d-18                  ! HI cross-section at HeI ionization frequency
   real(kind=real64), parameter :: sigma_H_at_HeII = 1.230695924714239d-19     ! HI cross-section at HeII ionization frequency
   real(kind=real64), parameter :: sigma_H_at_HeLya = 9.907d-22                ! HI cross-section at HeI Lya frequency (h\nu = 40.8 eV)
   real(kind=real64), parameter :: sigma_HeI_at_ion_freq = 7.430d-18           ! HeI cross section at its ionzing frequency
   real(kind=real64), parameter :: sigma_HeI_at_HeII = 1.690780687052975d-18   ! HeI cross-section at HeII ionization threshold
   real(kind=real64), parameter :: sigma_HeI_at_HeLya = 1.301d-20              ! HeI cross-section at HeI Lya frequency (h\nu = 40.8 eV)
   real(kind=real64), parameter :: sigma_HeII_at_ion_freq = 1.589d-18          ! HeII cross section at its ionzing frequency

   ! cosmological abundance
   real(kind=real64), parameter :: abu_he = 0.074_real64
   real(kind=real64), parameter :: abu_h = 0.926_real64
   real(kind=real64), parameter :: abu_c = 7.1d-7

   ! constants for thermal evolution
   real(kind=real64), parameter :: gamma = 5.0_real64/3.0_real64   ! monoatomic gas heat capacity ratio

contains

   function expm1x(x)

      implicit none
      real(kind=real64) :: expm1x
      real(kind=real64), intent(in) :: x

      if (abs(x) .lt. 1.0d-8) then
         expm1x = 1.0_real64
      else
         expm1x = expm1(x)/x
      end if

   end function expm1x

   function check_convergence(new, old)

      implicit none
      logical(kind=4) :: check_convergence
      real(kind=real64), intent(in) :: new, old

      check_convergence = (abs(new - old) < minimum_fractional_change*new) .or. &
                          (new < minimum_fraction_of_atoms)

   end function check_convergence

   function check_divergence(new, old)

      implicit none
      logical(kind=4) :: check_divergence
      real(kind=real64), intent(in) :: new, old
      real(kind=real64) :: diff

      diff = abs(new - old)
      check_divergence = (diff > minimum_fractional_change) .and. &
                         (diff > minimum_fractional_change*new) .and. &
                         (new > minimum_fraction_of_atoms)

   end function check_divergence

   subroutine thermal(dt, end_temper, avg_temper, ndens_el, ndens_atom, heat, Hz, cosmo_only)
      ! This is the wrapper that appears in chemistry_he
      real(real64), intent(in)    :: dt, ndens_el, ndens_atom, heat, Hz
      real(real64), intent(inout) :: end_temper, avg_temper
      logical(kind=4), intent(in) :: cosmo_only

      call thermal_impl(dt, end_temper, avg_temper, ndens_el, ndens_atom, heat, Hz, cosmo_only)
   end subroutine thermal

   ! TODO: pass the column density to global
   subroutine global_pass(dt, Hz, ndens, temp, temp_av, &
                          xHII, xHII_av, xHII_intermed, &
                          xHeII, xHeII_av, xHeII_intermed, &
                          xHeIII, xHeIII_av, xHeIII_intermed, &
                          phi_HI_ion, phi_HeI_ion, phi_HeII_ion, &
                          heat_HI_ion, heat_HeI_ion, heat_HeII_ion, &
                          clump, cosmo_only, conv_flag, m1, m2, m3)
      ! Subroutine Arguments
      real(kind=real64), intent(in) :: dt                         ! time step
      real(kind=real64), intent(in) :: Hz                         ! Hubble function at the corresponding redshift (in cgs)
      real(kind=real64), intent(in) :: temp(m1, m2, m3)             ! Temperature field
      real(kind=real64), intent(in) :: temp_av(m1, m2, m3)             ! time-averaged Temperature field
      real(kind=real64), intent(in) :: ndens(m1, m2, m3)            ! Gas density field
      real(kind=real64), intent(inout) :: xHII(m1, m2, m3)             ! HI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHII_av(m1, m2, m3)          ! Time-averaged HI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHII_intermed(m1, m2, m3)    ! Intermediate HI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeII(m1, m2, m3)             ! HeI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeII_av(m1, m2, m3)          ! Time-averaged HeI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeII_intermed(m1, m2, m3)    ! Intermediate HeI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeIII(m1, m2, m3)            ! HeII ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeIII_av(m1, m2, m3)         ! Time-averaged HeII ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeIII_intermed(m1, m2, m3)   ! Intermediate HeII ionization fractions of the cells
      real(kind=real64), intent(in) :: phi_HI_ion(m1, m2, m3)          ! HI Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
      real(kind=real64), intent(in) :: phi_HeI_ion(m1, m2, m3)         ! HeI Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
      real(kind=real64), intent(in) :: phi_HeII_ion(m1, m2, m3)        ! HeII Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
      real(kind=real64), intent(in) :: heat_HI_ion(m1, m2, m3)         ! HI Photo-heating rate for the whole grid
      real(kind=real64), intent(in) :: heat_HeI_ion(m1, m2, m3)        ! HeI Photo-heating rate for the whole grid
      real(kind=real64), intent(in) :: heat_HeII_ion(m1, m2, m3)       ! HeII Photo-heating rate for the whole grid
      real(kind=real64), intent(in) :: clump(m1, m2, m3)            ! Clumping factor field (even if it's just a constant it has to be a 3D cube)
      logical(kind=4), intent(in) :: cosmo_only    ! Whether to only include cosmological cooling

      integer, intent(in) :: m1                                   ! mesh size x (hidden by f2py)
      integer, intent(in) :: m2                                   ! mesh size y (hidden by f2py)
      integer, intent(in) :: m3                                   ! mesh size z (hidden by f2py)

      integer, intent(out) :: conv_flag

      integer :: i, j, k  ! mesh position

      ! Mesh position of the cell being treated
      integer, dimension(3) :: pos

      conv_flag = 0
      do k = 1, m3
         do j = 1, m2
            do i = 1, m1
               pos = (/i, j, k/)
               call evolve0D_global(dt, Hz, pos, ndens, temp, temp_av, &
                                    xHII, xHII_av, xHII_intermed, &
                                    xHeII, xHeII_av, xHeII_intermed, &
                                    xHeIII, xHeIII_av, xHeIII_intermed, &
                                    phi_HI_ion, phi_HeI_ion, phi_HeII_ion, &
                                    heat_HI_ion, heat_HeI_ion, heat_HeII_ion, &
                                    clump, cosmo_only, conv_flag, m1, m2, m3)
            end do
         end do
      end do

   end subroutine global_pass

   subroutine evolve0D_global(dt, Hz, pos, ndens, temp, temp_av, &
                              xHII, xHII_av, xHII_intermed, &
                              xHeII, xHeII_av, xHeII_intermed, &
                              xHeIII, xHeIII_av, xHeIII_intermed, &
                              phi_HI_ion, phi_HeI_ion, phi_HeII_ion, &
                              heat_HI_ion, heat_HeI_ion, heat_HeII_ion, &
                              clump, cosmo_only, conv_flag, m1, m2, m3)
      ! Subroutine Arguments
      real(kind=real64), intent(in) :: dt                         ! time step
      real(kind=real64), intent(in) :: Hz                         ! Hubble function at the corresponding redshift (in cgs)
      integer, dimension(3), intent(in) :: pos                      ! cell position
      real(kind=real64), intent(in) :: temp(m1, m2, m3)             ! Temperature field
      real(kind=real64), intent(in) :: temp_av(m1, m2, m3)             ! time-averaged Temperature field
      real(kind=real64), intent(in) :: ndens(m1, m2, m3)            ! Hydrogen Density Field
      real(kind=real64), intent(inout) :: xHII(m1, m2, m3)             ! HI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHII_av(m1, m2, m3)          ! Time-averaged HI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHII_intermed(m1, m2, m3)    ! Intermediate HI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeII(m1, m2, m3)             ! HeI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeII_av(m1, m2, m3)          ! Time-averaged HeI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeII_intermed(m1, m2, m3)    ! Intermediate HeI ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeIII(m1, m2, m3)             ! HeII ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeIII_av(m1, m2, m3)          ! Time-averaged HeII ionization fractions of the cells
      real(kind=real64), intent(inout) :: xHeIII_intermed(m1, m2, m3)    ! Intermediate HeII ionization fractions of the cells
      real(kind=real64), intent(in) :: phi_HI_ion(m1, m2, m3)           ! H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
      real(kind=real64), intent(in) :: phi_HeI_ion(m1, m2, m3)          ! HeI Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
      real(kind=real64), intent(in) :: phi_HeII_ion(m1, m2, m3)         ! HeII Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
      real(kind=real64), intent(in) :: heat_HI_ion(m1, m2, m3)          ! HI Photo-heating rate for the whole grid
      real(kind=real64), intent(in) :: heat_HeI_ion(m1, m2, m3)         ! HeI Photo-heating rate for the whole grid
      real(kind=real64), intent(in) :: heat_HeII_ion(m1, m2, m3)        ! HeII Photo-heating rate for the whole grid
      real(kind=real64), intent(in) :: clump(m1, m2, m3)             ! Clumping factor field (even if it's just a constant it has to be a 3D cube)
      logical(kind=4), intent(in) :: cosmo_only   ! Whether to only include cosmological cooling

      integer, intent(inout) :: conv_flag                          ! convergence counter
      integer, intent(in) :: m1                                   ! mesh size x (hidden by f2py)
      integer, intent(in) :: m2                                   ! mesh size y (hidden by f2py)
      integer, intent(in) :: m3                                   ! mesh size z (hidden by f2py)

      ! Local quantities
      real(kind=real64) :: temperature_start
      real(kind=real64) :: ndens_p                        ! local gas density
      real(kind=real64) :: xHII_p, xHeII_p, xHeIII_p          ! local hydrogen ionization fraction
      real(kind=real64) :: xHII_av_p, xHeII_av_p, xHeIII_av_p ! local hydrogen  mean ionization fraction
      real(kind=real64) :: xHII_intermed_p, xHeII_intermed_p, xHeIII_intermed_p! local hydrogen mean ionization fraction
      real(kind=real64) :: yh_av_p        ! local mean neutral fraction TODO: do we still need it? also for He then?
      real(kind=real64) :: phi_HI_ion_p, phi_HeI_ion_p, phi_HeII_ion_p    ! local photo-ionization rate
      real(kind=real64) :: heat_HI_ion_p, heat_HeI_ion_p, heat_HeII_ion_p    ! local photo-heating rate
      real(kind=real64) :: coldend_HI_p, coldend_HeI_p, coldend_HeII_p    ! local photo-heating rate
      real(kind=real64) :: xHII_av_p_old, xHeII_av_p_old, xHeIII_av_p_old     ! mean ion fraction before chemistry (to check convergence)
      real(kind=real64) :: clump_p        ! local clumping factor

      ! Initialize local quantities
      temperature_start = temp(pos(1), pos(2), pos(3))
      ndens_p = ndens(pos(1), pos(2), pos(3))
      phi_HI_ion_p = phi_HI_ion(pos(1), pos(2), pos(3))
      phi_HeI_ion_p = phi_HeI_ion(pos(1), pos(2), pos(3))
      phi_HeII_ion_p = phi_HeII_ion(pos(1), pos(2), pos(3))
      heat_HI_ion_p = heat_HI_ion(pos(1), pos(2), pos(3))
      heat_HeI_ion_p = heat_HeI_ion(pos(1), pos(2), pos(3))
      heat_HeII_ion_p = heat_HeII_ion(pos(1), pos(2), pos(3))
      clump_p = clump(pos(1), pos(2), pos(3))

      ! Initialize local ion fractions
      xHII_p = xHII(pos(1), pos(2), pos(3))
      xHII_av_p = xHII_av(pos(1), pos(2), pos(3))
      xHII_intermed_p = xHII_intermed(pos(1), pos(2), pos(3))
      xHeII_p = xHeII(pos(1), pos(2), pos(3))
      xHeII_av_p = xHeII_av(pos(1), pos(2), pos(3))
      xHeII_intermed_p = xHeII_intermed(pos(1), pos(2), pos(3))
      xHeIII_p = xHeIII(pos(1), pos(2), pos(3))
      xHeIII_av_p = xHeIII_av(pos(1), pos(2), pos(3))
      xHeIII_intermed_p = xHeIII_intermed(pos(1), pos(2), pos(3))
      !yh_av_p = 1.0 - xHII_av_p

      call do_chemistry(dt, Hz, ndens_p, temperature_start, &
                        xHII_p, xHII_av_p, xHII_intermed_p, &
                        xHeII_p, xHeII_av_p, xHeII_intermed_p, &
                        xHeIII_p, xHeIII_av_p, xHeIII_intermed_p, &
                        phi_HI_ion_p, phi_HeI_ion_p, phi_HeII_ion_p, &
                        heat_HI_ion_p, heat_HeI_ion_p, heat_HeII_ion_p, &
                        clump_p, cosmo_only)

      ! Check for convergence (global flag). In original, convergence is tested using neutral fraction, but testing with ionized fraction should be equivalent.
      ! TODO: add temperature convergence criterion when non-isothermal mode is added later on.
      xHII_av_p_old = xHII_av(pos(1), pos(2), pos(3))
      xHeII_av_p_old = xHeII_av(pos(1), pos(2), pos(3))
      xHeIII_av_p_old = xHeIII_av(pos(1), pos(2), pos(3))

      ! Hydrogen criterion
      if (check_divergence(1.0_real64 - xHII_av_p, 1.0_real64 - xHII_av_p_old) .or. &
          check_divergence(1.0_real64 - xHII_av_p - xHeII_av_p, 1.0_real64 - xHII_av_p_old - xHeII_av_p_old) .or. &
          check_divergence(xHeIII_av_p, xHeIII_av_p_old)) then
         ! TODO: missing temperature check
         conv_flag = conv_flag + 1
      end if

      ! Put local result in global array
      xHII_intermed(pos(1), pos(2), pos(3)) = xHII_intermed_p
      xHII_av(pos(1), pos(2), pos(3)) = xHII_av_p

      xHeII_intermed(pos(1), pos(2), pos(3)) = xHeII_intermed_p
      xHeII_av(pos(1), pos(2), pos(3)) = xHeII_av_p

      xHeIII_intermed(pos(1), pos(2), pos(3)) = xHeIII_intermed_p
      xHeIII_av(pos(1), pos(2), pos(3)) = xHeIII_av_p

   end subroutine evolve0D_global

   ! ===============================================================================================
   ! Adapted version of do_chemistry that excludes the "local" part (which is effectively unused in
   ! the current version of c2ray). This subroutine takes grid-arguments along with a position.
   ! Original: G. Mellema (2005)
   ! This version: P. Hirling (2023)
   ! ===============================================================================================
   subroutine do_chemistry(dt, Hz, ndens_p, temperature_start, &
                           xHII_p, xHII_av_p, xHII_intermed_p, &
                           xHeII_p, xHeII_av_p, xHeII_intermed_p, &
                           xHeIII_p, xHeIII_av_p, xHeIII_intermed_p, &
                           phi_HI_ion_p, phi_HeI_ion_p, phi_HeII_ion_p, &
                           heat_HI_ion_p, heat_HeI_ion_p, heat_HeII_ion_p, &
                           clump_p, cosmo_only)
      ! Subroutine Arguments
      real(kind=real64), intent(in) :: dt                    ! time step
      real(kind=real64), intent(in) :: Hz                    ! Hubble function at the corresponding redshift (in cgs)
      real(kind=real64), intent(in) :: temperature_start    ! Local starting temperature
      real(kind=real64), intent(in) :: ndens_p              ! Local gas number density (cgs)
      real(kind=real64), intent(inout) :: xHII_p, xHeII_p, xHeIII_p              ! HII, HeII, and HeIII ionization fractions of the cells
      real(kind=real64), intent(out) :: xHII_av_p, xHeII_av_p, xHeIII_av_p            ! HII, HeII, and HeIII time-averaged ionization fractions of the cells
      real(kind=real64), intent(out) :: xHII_intermed_p, xHeII_intermed_p, xHeIII_intermed_p  ! intermediate HII, HeII, and HeIII ionization fractions of the cells
      real(kind=real64), intent(in) :: phi_HI_ion_p, phi_HeI_ion_p, phi_HeII_ion_p        ! Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
      real(kind=real64), intent(in) :: heat_HI_ion_p, heat_HeI_ion_p, heat_HeII_ion_p     ! Photo-heating rate for the whole grid
      real(kind=real64), intent(in) :: clump_p             ! Local clumping factor
      !real(kind=real64), intent(in) :: abu_c                 ! Carbon abundance
      logical(kind=4), intent(in) :: cosmo_only    ! Whether to only include cosmological cooling

      ! Local quantities
      real(kind=real64) :: xHII_av_p_2, xHeII_av_p_2, xHeIII_av_p_2
      real(kind=real64) :: xHII_intermed_p_2, xHeII_intermed_p_2, xHeIII_intermed_p_2
      real(kind=real64) :: nHI_p, nHeI_p, nHeII_p            ! HI, HeI, and HeII number density
      real(kind=real64) :: heating                              ! total heating rate
      real(kind=real64) :: coldhi_p, coldhei_p, coldheii_p      ! column density of the cell for the three spicies
      real(kind=real64) :: temperature_avg, temperature_int
      real(kind=real64) :: temperature_avg_new, temperature_int_new
      real(kind=real64) :: xHII_av_p_new, xHeII_av_p_new, xHeIII_av_p_new   ! Time-average ionization fraction from previous iteration
      real(kind=real64) :: de                               ! local electron density
      real(kind=real64) :: xHI_new, xHI
      integer :: nit                                        ! Iteration counter

      ! Temperature at the begin of the while loop
      temperature_int = temperature_start
      temperature_avg = temperature_start

      ! Total heating rate from the three components
      heating = heat_HI_ion_p + heat_HeI_ion_p + heat_HeII_ion_p

      nit = 0
      do
         nit = nit + 1

         ! At each iteration, the intial condition x(0) is reset. Change happens in the time-average and thus the electron density
         ! Calculate (mean) elements density
         nHI_p = ndens_p*abu_h*(1.0_real64 - xHII_intermed_p)
         nHeI_p = ndens_p*abu_he*(1.0_real64 - xHeII_intermed_p - xHeIII_intermed_p)
         nHeII_p = ndens_p*abu_he*xHeII_intermed_p

         ! Calculate (mean) electron density
         de = electrondens(ndens_p, xHII_av_p, xHeII_av_p, xHeIII_av_p, abu_h, abu_he, abu_c)

         ! TODO: collisional ionisation
         ! call ini_rec_colion_factors(temperature_previous_iteration)

         ! Calculate the new and mean ionization states
         ! TODO: the intermediate need in the python evolve.py for global convergence. Keep it and bring it back.
         call friedrich(dt, temperature_avg, de, &
                        xHII_p, xHeII_p, xHeIII_p, &
                        phi_HI_ion_p, phi_HeI_ion_p, phi_HeII_ion_p, &
                        heat_HI_ion_p, heat_HeI_ion_p, heat_HeII_ion_p, &
                        nHI_p, nHeI_p, nHeII_p, clump_p, &
                        xHII_intermed_p, xHeII_intermed_p, xHeIII_intermed_p, &
                        xHII_av_p_new, xHeII_av_p_new, xHeIII_av_p_new)

         ! update (mean) electron density after updating averaged quantities
         de = electrondens(ndens_p, xHII_av_p_new, xHeII_av_p_new, xHeIII_av_p_new, abu_h, abu_he, abu_c)

         nHI_p = ndens_p*abu_h*(1.0_real64 - xHII_intermed_p)
         nHeI_p = ndens_p*abu_he*(1.0_real64 - xHeII_intermed_p - xHeIII_intermed_p)
         nHeII_p = ndens_p*abu_he*xHeII_intermed_p

         call friedrich(dt, temperature_avg, de, &
                        xHII_intermed_p, xHeII_intermed_p, xHeIII_intermed_p, &
                        phi_HI_ion_p, phi_HeI_ion_p, phi_HeII_ion_p, &
                        heat_HI_ion_p, heat_HeI_ion_p, heat_HeII_ion_p, &
                        nHI_p, nHeI_p, nHeII_p, clump_p, &
                        xHII_intermed_p_2, xHeII_intermed_p_2, xHeIII_intermed_p_2, &
                        xHII_av_p_2, xHeII_av_p_2, xHeIII_av_p_2)

         ! Average two solutions
         xHII_intermed_p = 0.5_real64*(xHII_intermed_p + xHII_intermed_p_2)
         xHeII_intermed_p = 0.5_real64*(xHeII_intermed_p + xHeII_intermed_p_2)
         xHeIII_intermed_p = 0.5_real64*(xHeIII_intermed_p + xHeIII_intermed_p_2)
         xHII_av_p_new = 0.5_real64*(xHII_av_p_new + xHII_av_p_2)
         xHeII_av_p_new = 0.5_real64*(xHeII_av_p_new + xHeII_av_p_2)
         xHeIII_av_p_new = 0.5_real64*(xHeIII_av_p_new + xHeIII_av_p_2)

         ! update (mean) electron density after updating averaged quantities
         de = electrondens(ndens_p, xHII_av_p_new, xHeII_av_p_new, xHeIII_av_p_new, abu_h, abu_he, abu_c)

         ! TODO: Call for thermal evolution. It takes the old values and outputs new values without overwriting the old values.
         temperature_int_new = temperature_start
         call thermal(dt, temperature_int_new, temperature_avg_new, de, ndens_p, heating, Hz, cosmo_only)

         ! TODO: multiphase is necessary to correctly calculate the differantial brightness. Hannah's works is on github with helium: https://github.com/garrelt/C2-Ray3Dm1D_Helium/blob/multiphase/code/files_for_3D/evolve_data.F90#L37-L39

         xHI_new = 1.0_real64 - xHII_av_p_new
         xHI = 1.0_real64 - xHII_av_p
         ! Test for convergence on time-averaged neutral fraction. For low values of this number assume convergence
         if (check_convergence(xHI_new, xHI) .and. &
             check_convergence(xHI_new - xHeII_av_p_new, xHI - xHeII_av_p) .and. &
             check_convergence(xHeIII_av_p_new, xHeIII_av_p) .and. &
             (abs((temperature_int - temperature_int_new)/temperature_int_new) < minimum_fractional_change)) then
            nit = 500 ! Exit loop
         end if

         xHII_av_p = xHII_av_p_new
         xHeII_av_p = xHeII_av_p_new
         xHeIII_av_p = xHeIII_av_p_new

         temperature_int = temperature_int_new
         temperature_avg = temperature_avg_new

         ! Warn about non-convergence and terminate iteration
         if (nit > 400) then
            exit
         end if
      end do
   end subroutine do_chemistry

   ! ===============================================================================================
   ! Calculates time dependent ionization state for hydrogen and helium
   ! Author: Martina Friderich (2012)
   ! 1 November 2024: adapted for f2py (M. Bianco)
   !
   ! Adapted version of Friderich+ (2012) method as an extension to the Altay+ (2008) analytical solution.
   ! I employed Kai Yan Lee PhD thesis as reference. The naming of variables changed a bit compared to Martina's code and istead I adopted the naming system of the equations in Kai's thesis. However, be carefull because Kai Yan Lee's thesis has a mistake in equation (2.61) when compared to Friderich+ (2012) equation (B8). In that case I followed the latter.
   ! ===============================================================================================
   subroutine friedrich(dt, temp_p, n_e, &
                        xHII_old, xHeII_old, xHeIII_old, &
                        phi_HI, phi_HeI, phi_HeII, heat_HI, heat_HeI, heat_HeII, &
                        nHI_p, nHeI_p, nHeII_p, clumping, &
                        xHII, xHeII, xHeIII, &
                        xHII_av, xHeII_av, xHeIII_av)

      ! Input & output arguments
      real(kind=real64), intent(in) :: dt                                 ! time step and cell size (cgs)
      real(kind=real64), intent(in) :: xHII_old, xHeII_old, xHeIII_old    ! previous ionized fractions
      real(kind=real64), intent(in) :: temp_p, n_e                        ! local temperature and electron number density
      real(kind=real64), intent(in) :: phi_HI, phi_HeI, phi_HeII          ! photo-ionization rates for the three species
      real(kind=real64), intent(in) :: heat_HI, heat_HeI, heat_HeII       ! photo-heating rates for the three species
      real(kind=real64), intent(in) :: nHI_p, nHeI_p, nHeII_p             ! cell element density (cgs)
      real(kind=real64), intent(in) :: clumping                           ! local clumping factor
      real(kind=real64), intent(out) :: xHII, xHeII, xHeIII               ! analytical solution for the ionized fractions
      real(kind=real64), intent(out) :: xHII_av, xHeII_av, xHeIII_av      ! averaged solution for the ionized fractions
      real(kind=real64) :: xHeI, xHeI_av, norm

      ! Local variables for Doric methods
      real(kind=real64) :: alphA_HII, alphB_HII, alph1_HII
      real(kind=real64) :: alphA_HeII, alphB_HeII, alph1_HeII
      real(kind=real64) :: alphA_HeIII, alphB_HeIII, alph1_HeIII, alph2_HeIII
      real(kind=real64) :: nu
      real(kind=real64) :: tau_H_at_HeI, tau_HeI_at_ion_freq, tau_H_at_HeLya, tau_He_at_HeLya
      real(kind=real64) :: tau_H_at_HeII, tau_HeII_at_ion_freq, tau_HeI_at_HeII
      real(kind=real64) :: yy, zz, y2a, y2b
      real(kind=real64) :: cHI, cHeI, cHeII, uHI, uHeI, uHeII
      real(kind=real64) :: rHII_HI, rHeII_HI, rHeII_HeI, rHeIII_HI, rHeIII_HeI, rHeIII_HeII
      real(kind=real64) :: S, K, R, T, lamb1, lamb2, lamb3
      real(kind=real64) :: A11, A12, A13, A22, A23, A32, A33
      real(kind=real64) :: x1_1, x2_1, x3_1, x2_2, x3_2, x2_3, x3_3
      real(kind=real64) :: c1, c2, c3, p1, p2, p3
      real(kind=real64) :: el1, el2, el3
      real(kind=real64) :: lambda, dielec
      real(kind=real64) :: f_lya       ! "escape” fraction of Ly α photons, it depends on the neutral fraction
      real(kind=real64), parameter :: p_rec = 0.96_real64
      real(kind=real64), parameter :: l_dec = 1.425_real64
      real(kind=real64), parameter :: m_dec = 0.737_real64
      real(kind=real64), parameter  :: ev2k = 1.0_real64/8.617d-05
      real(kind=real64), parameter :: temph0 = 13.598_real64*ev2k
      real(kind=real64), parameter :: temphe0 = 24.587_real64*ev2k
      real(kind=real64), parameter :: temphe1 = 54.416_real64*ev2k
      real(kind=real64), parameter :: colh0 = 1.3d-8*0.83_real64/(13.598_real64**2)
      real(kind=real64), parameter :: colhe0 = 1.3d-8*1.26_real64/(24.587_real64**2)
      real(kind=real64), parameter :: colhe1 = 1.3d-8*1.30_real64/(54.416_real64**2)
      real(kind=real64), parameter :: epsilon = 1.0d-20

      f_lya = max(min(10.0_real64*(1.0_real64 - xHII), 1.0_real64), 0.01_real64)

      ! Recombination rate of HI (Eq. 2.12 and 2.13)
      lambda = 2.0_real64*(temph0/temp_p)
      alphA_HII = 1.269d-13*lambda**1.503_real64/(1.0_real64 + (lambda/0.522_real64)**0.470_real64)**1.923_real64
      alphB_HII = 2.753d-14*lambda**1.500_real64/(1.0_real64 + (lambda/2.740_real64)**0.407_real64)**2.242_real64
      alph1_HII = alphA_HII - alphB_HII ! UNUSED

      ! Recombination rate of HeII (Eq. 2.14-17)
      if (temp_p < 9.0d3) then
         alphA_HeII = alphA_HII
         alphB_HeII = alphB_HII
      else
         lambda = 2.0_real64*(temphe0/temp_p)
         dielec = 1.9d-3*temp_p**(-1.5_real64)*exp(-4.7d5/temp_p)*(1.0_real64 + 0.3_real64*exp(-9.4d4/temp_p))
         alphA_HeII = 3.000d-14*lambda**0.654_real64 + dielec
         alphB_HeII = 1.260d-14*lambda**0.750_real64 + dielec
      end if
      alph1_HeII = alphA_HeII - alphB_HeII

      ! Recombination rate of HeIII (Eq. 2.18-20)
      lambda = 2.0_real64*(temphe1/temp_p)
      alphA_HeIII = 2.538d-13*lambda**1.503_real64/(1.0_real64 + (lambda/0.522_real64)**0.470_real64)**1.923_real64
      alphB_HeIII = 5.5060d-14*lambda**1.5_real64/(1.0_real64 + (lambda/2.740_real64)**0.407_real64)**2.242_real64
      alph1_HeIII = alphA_HeIII - alphB_HeIII     ! this was not specified in Kay Yan Lee thesis, but confirmed by Garrelt (13.10.24)
      alph2_HeIII = 3.4d-13*(temp_p/1.0d4)**(-0.6_real64)  ! extrapolate Osterbrok B value minus C value, p 38

      ! two photons emission from recombination of HeIII
      nu = 0.285_real64*(temp_p/1.0d4)**0.119_real64

      ! Comoving distance dr is factorized out from the optical depth calculation because it does not affect the ratios.
      ! optical depth of HI at HeI ionation frequency threshold
      tau_H_at_HeI = nHI_p*sigma_H_at_HeI

      ! optical depth of HeI at HeI ionation frequency threshold
      tau_HeI_at_ion_freq = nHeI_p*sigma_HeI_at_ion_freq

      ! optical depth of H and He at he+Lya (40.817eV)
      tau_H_at_HeLya = nHI_p*sigma_H_at_HeLya
      tau_He_at_HeLya = nHeI_p*sigma_HeI_at_HeLya

      ! optical depth of H at HeII ion threshold
      tau_H_at_HeII = nHI_p*sigma_H_at_HeII

      ! optical depth of HeI at HeII ion threshold
      tau_HeI_at_HeII = nHeI_p*sigma_HeI_at_HeII

      ! optical depth of HeII at HeII ion threshold
      tau_HeII_at_ion_freq = nHeII_p*sigma_HeII_at_ion_freq

      ! Ratios of these optical depths needed in doric
      yy = tau_H_at_HeI/(tau_H_at_HeI + tau_HeI_at_ion_freq)
      zz = tau_H_at_HeLya/(tau_H_at_HeLya + tau_He_at_HeLya)
      y2a = tau_HeII_at_ion_freq/(tau_H_at_HeII + tau_HeI_at_HeII + tau_HeII_at_ion_freq)
      y2b = tau_HeI_at_HeII/(tau_H_at_HeII + tau_HeI_at_HeII + tau_HeII_at_ion_freq)

      ! Collisional ionization process (Eq. 2.21-23)
      ! TODO: a remarks is that in principle collisional ionization is also clumping dependent (but HI clumping) but probably irrelevant at this scale.
      cHI = colh0*sqrt(temp_p)*exp(-temph0/temp_p)
      cHeI = colhe0*sqrt(temp_p)*exp(-temphe0/temp_p)
      cHeII = colhe1*sqrt(temp_p)*exp(-temphe1/temp_p)

      ! Photo-ionization rates (Eq. 2.27-29)
      uHI = max(phi_HI + cHI*n_e, 1d-200)
      uHeI = max(phi_HeI + cHeI*n_e, 1d-200)
      uHeII = max(phi_HeII + cHeII*n_e, 1d-200)

      ! Recombination rate (Eq. 2.30-35)
      rHII_HI = -alphB_HII
      rHeII_HI = p_rec*alphB_HeII + yy*alph1_HeII
      rHeIII_HI = (1 - y2a - y2b)*alph1_HeIII + alph2_HeIII + (nu*(l_dec - m_dec + m_dec*yy) + (1 - nu)*f_lya*zz)*alphB_HeIII

      rHeII_HeI = (1 - yy)*alph1_HeII - alphA_HeII
      rHeIII_HeI = (y2b - y2a)*alph1_HeIII + (nu*m_dec*(1 - yy) + f_lya*(1 - nu)*(1 - zz))*alphB_HeIII + alphA_HeIII

      rHeIII_HeII = y2a*alph1_HeIII - alphA_HeIII

      ! get matrix elements
      A11 = -uHI + rHII_HI*n_e
      A12 = abu_he/abu_h*n_e*rHeII_HI
      A13 = abu_he/abu_h*n_e*rHeIII_HI
      !A21 = 0.0
      A22 = -uHeI - uHeII + rHeII_HeI*n_e
      A23 = -uHeI + rHeIII_HeI*n_e
      !A31 = 0.0
      A32 = uHeII
      A33 = rHeIII_HeII*n_e

      ! define coefficients
      S = sqrt((A33 - A22)**2.0 + 4.0*A32*A23)
      K = 1.0_real64/(A32*A23 - A33*A22)
      R = 2.0_real64*A32*(A33*uHeI*K - xHeII_old)
      T = -A32*uHeI*K - xHeIII_old

      ! Define eigen-value
      lamb1 = A11
      lamb2 = (A33 + A22 - S)/2.0
      lamb3 = (A33 + A22 + S)/2.0

      ! Particular solution vector
      p1 = -(uHI + (A33*A12 - A32*A13)*uHeI*K)/A11
      p2 = A33*uHeI*K
      p3 = -A32*uHeI*K

      ! Define the eigen vectors
      ! eigen vector x1
      !x1_1 = 1.0_real64
      !x1_2 = 0.0
      !x1_3 = 0.0
      ! eigen vector x2
      x2_1 = (-2.0_real64*A32*A13 + A12*(A33 - A22 + S))/(2.0_real64*A32*(A11 - lamb2))
      x2_2 = (-A33 + A22 - S)/(2.0_real64*A32)
      !x2_3 = 1.0_real64
      ! eigen vector x3
      x3_1 = (-2.0_real64*A32*A13 + A12*(A33 - A22 - S))/(2.0_real64*A32*(A11 - lamb3))
      x3_2 = (-A33 + A22 + S)/(2.0_real64*A32)
      !x3_3 = 1.0_real64

      ! Coefficients from boundary conditions
      ! This formulation keeps numeric error low
      c1 = -p1 + xHII_old + ((R + T*(A33 - A22 + S))*x3_1 - (R + T*(A33 - A22 - S))*x2_1)/(2.0*S)
      c2 = (R + (A33 - A22 - S)*T)/(2.0*S)
      c3 = -(R + (A33 - A22 + S)*T)/(2.0*S)

      el1 = exp(lamb1*dt)
      el2 = exp(lamb2*dt)
      el3 = exp(lamb3*dt)

      ! Define analytical solution (Eq. 2.39-41)
      xHII = p1 + c1*el1 + x2_1*c2*el2 + x3_1*c3*el3
      xHeII = p2 + x2_2*c2*el2 + x3_2*c3*el3
      xHeIII = p3 + c2*el2 + c3*el3

      el1 = expm1x(lamb1*dt)
      el2 = expm1x(lamb2*dt)
      el3 = expm1x(lamb3*dt)

      ! Define time average solution (Eq. 2.64-68)
      xHII_av = p1 + c1*el1 + x2_1*c2*el2 + x3_1*c3*el3
      xHeII_av = p2 + x2_2*c2*el2 + x3_2*c3*el3
      xHeIII_av = p3 + c2*el2 + c3*el3

      ! Fix numerical limits and renormalize
      xHeI = 1.0 - xHeII - xHeIII
      xHII = min(max(epsilon, xHII), 1.0_real64)
      xHeII = min(max(epsilon, xHeII), 1.0_real64)
      xHeIII = min(max(epsilon, xHeIII), 1.0_real64)
      norm = xHeI + xHeII + xHeIII
      xHeII = xHeII/norm
      xHeIII = xHeIII/norm

      ! Fix numerical limits and renormalize
      xHeI_av = 1.0 - xHeII_av - xHeIII_av
      xHII_av = min(max(epsilon, xHII_av), 1.0_real64)
      xHeII_av = min(max(epsilon, xHeII_av), 1.0_real64)
      xHeIII_av = min(max(epsilon, xHeIII_av), 1.0_real64)
      norm = xHeI_av + xHeII_av + xHeIII_av
      xHeII_av = xHeII_av/norm
      xHeIII_av = xHeIII_av/norm

   end subroutine friedrich

end module chemistry_he
