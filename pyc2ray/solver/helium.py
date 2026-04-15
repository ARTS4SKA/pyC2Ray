import warnings

import pyc2ray.constants as c


def get_temperature(energy: float, ndens: float, gamma: float = 5 / 3) -> float:
    """Return the temperature (K) for a given internal energy per unit volume and number density."""
    return energy * (gamma - 1.0) / (c.k_B * ndens)


def get_energy(temp: float, ndens: float, gamma: float = 5 / 3) -> float:
    """Return the internal energy per unit volume (erg/cm^3) for a given temperature and number density."""
    return temp * (c.k_B * ndens) / (gamma - 1.0)


def cooling_rate(n_a: float, n_e: float, temp: float) -> float:
    """Return the cooling rate per unit volume (erg/s/cm^3) for a given atomic density, electron density, and temperature."""
    warnings.warn(
        "This function is a placeholder and should be replaced with a real cooling function.",
        DeprecationWarning,
    )
    return n_a * n_e * temp


def cosmo_cooling_rate(energy: float, Hz: float) -> float:
    """Return the cosmological cooling rate per unit volume (erg/s/cm^3) for a given internal energy and Hubble parameter."""
    return 2.0 * energy * Hz


def thermal(
    dt: float,
    start_temp: float,
    ndens_e: float,
    ndens_a: float,
    heating: float,
    Hz: float,
    relative_denergy: float = 0.1,
    gamma: float = 5.0 / 3.0,
    min_temp: float = 1.0,
    cosmo_only: bool = True,
    max_iterations: int = 10000,
) -> tuple[float, float]:
    """Evolve the temperature of a gas parcel over a time step dt, given initial conditions and heating/cooling rates.
    Parameters
    ----------
    dt :
        Time step over which to evolve the temperature (s).
    start_temp :
        Initial temperature of the gas (K).
    ndens_e :
        Electron number density (cm^-3).
    ndens_a :
        Atomic number density (cm^-3).
    heating :
        Heating rate per unit volume (erg/s/cm^3).
    Hz :
        Hubble parameter at the current redshift (s^-1).
    relative_denergy :
        Maximum allowed relative change in internal energy per iteration, by default 0.1.
    gamma :
        Adiabatic index of the gas, by default 5/3.
    min_temp :
        Minimum allowed temperature (K), by default 1.0.
    cosmo_only :
        If True, only include cosmological cooling; if False, include both cosmological and atomic
        cooling, by default True.
    max_iterations :
        Maximum number of iterations to perform, by default 10000.
    """
    if start_temp <= min_temp:
        return start_temp, start_temp

    u0 = get_energy(start_temp, ndens_a + ndens_e, gamma)
    u_min = get_energy(min_temp, ndens_a + ndens_e, gamma)
    ui = u0
    ui_av = u0
    Ti = start_temp
    tot_time = 0.0

    niter = 0
    while niter < max_iterations and tot_time < dt * (1 - 1e-6):
        rate = heating - cosmo_cooling_rate(ui, Hz)
        if not cosmo_only:
            rate -= cooling_rate(ndens_a, ndens_e, Ti)
        subdt = min(relative_denergy * ui / abs(rate), dt - tot_time)

        ui += rate * subdt
        ui_av += rate * subdt**2 / dt
        Ti = get_temperature(ui, ndens_a + ndens_e, gamma)

        tot_time += subdt
        niter += 1
        if ui < u_min:
            ui = u_min
            Ti = min_temp
            break

    end_temp = Ti
    avg_temp = get_temperature(ui_av, ndens_a + ndens_e, gamma)

    return end_temp, avg_temp
