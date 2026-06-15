import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import pyc2ray.constants as c

PathType = str | os.PathLike


def get_temperature(energy: float, ndens: float, gamma: float = 5 / 3) -> float:
    """Return the temperature (K) for a given internal energy per unit volume and number density."""
    return energy * (gamma - 1.0) / (c.k_B * ndens)


def get_energy(temp: float, ndens: float, gamma: float = 5 / 3) -> float:
    """Return the internal energy per unit volume (erg/cm^3) for a given temperature and number density."""
    return temp * (c.k_B * ndens) / (gamma - 1.0)


@dataclass
class CoolingTables:
    HI: np.ndarray
    HII: np.ndarray
    HeI: np.ndarray
    HeII: np.ndarray
    HeIII: np.ndarray
    logtemp: tuple[float, float, int]
    tables_directory: PathType = Path(__file__).parent.parent / "tables" / "cooling"

    @classmethod
    def from_dir(
        cls,
        directory: None | PathType = None,
    ) -> "CoolingTables":
        """Load cooling tables from files in direcotry.

        Parameters
        ----------
        directory:
            path to the directory containing the cooling tables. If None, use the default directory in the package.
        """
        directory = Path(directory or cls.tables_directory)
        table_filenames = {
            "HI": directory / "HI_cool.txt",
            "HII": directory / "HII_coolB.txt",
            "HeI": directory / "HeI_cool.txt",
            "HeII": directory / "HeII_cool_nocollion.txt",
            "HeIII": directory / "HeIII_cool.txt",
        }

        logT, _ = np.loadtxt(table_filenames["HI"], unpack=True)
        logtemp = logT[0].item(), np.round(logT[1] - logT[0], 6).item(), len(logT)

        def load_table(filename: PathType) -> np.ndarray:
            data = np.pow(10, np.loadtxt(filename, unpack=True)[1])
            return np.insert(data, 0, 0.0)

        kwargs: dict[str, Any] = {x: load_table(f) for x, f in table_filenames.items()}
        kwargs["logtemp"] = logtemp
        kwargs["tables_directory"] = directory
        return cls(**kwargs)

    def astuple(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the cooling tables as a tuple."""
        return (
            self.HI,
            self.HII,
            self.HeI,
            self.HeII,
            self.HeIII,
        )


def cooling_rate(
    n_a: float,
    n_e: float,
    temp: float,
    xHI: float,
    xHeI: float,
    xHeII: float,
    tables: CoolingTables,
    abu_h: float,
    abu_he: float,
) -> float:
    """
    Parameters
    ----------
    nucldens:
        nuclei number density
    eldens:
        electron number density
    xhi:
        ionised fraction of the different species
    temp:
        temperature
    abu_h:
        abundance of H
    abu_he:
        abundance of He

    Returns
    -------
    rate: combined cooling rate in erg/s units
    """
    tstart, tstep, tnum = tables.logtemp

    # Find the position of the temperature in the table
    # NOTE(TB): Using the same interpolating formula as in rates.cu, which is different from MB notes from 20.05.26;
    # it instead reads: interp = min(tnum - 1, (ltemp - tstart) / tstep)
    ltemp = max(tstart, math.log10(temp))
    interp = (ltemp - tstart) / tstep
    p, r = math.modf(interp)
    q = 1 - p
    i0 = max(0, min(tnum - 1, int(r))) + 1
    i1 = i0 + 1

    xHII = 1.0 - xHI
    xHeIII = 1.0 - xHeI - xHeII

    # Combined cooling tables
    rHI = xHI * (tables.HI[i0] * q + tables.HI[i1] * p)
    rHII = xHII * (tables.HII[i0] * q + tables.HII[i1] * p)
    rHeI = xHeI * (tables.HeI[i0] * q + tables.HeI[i1] * p)
    rHeII = xHeII * (tables.HeII[i0] * q + tables.HeII[i1] * p)
    rHeIII = xHeIII * (tables.HeIII[i0] * q + tables.HeIII[i1] * p)

    rate = (rHI + rHII) * abu_h + (rHeI + rHeII + rHeIII) * abu_he
    return n_a * n_e * rate


def cosmo_cooling_rate(energy: float, Hz: float) -> float:
    """Return the cosmological cooling rate per unit volume (erg/s/cm^3)
    for a given internal energy and Hubble parameter."""
    return 2.0 * energy * Hz


def get_electron_density(
    ndens: float,
    xh: tuple[float, float, float],
    *,
    abu_h: float = 0.926,
    abu_he: float = 0.074,
    abu_c: float = 7.1e-7,
) -> float:
    """Calculate the electron number density from the atomic number density and ionized fractions."""
    xhii, xheii, xheiii = xh
    return ndens * (abu_h * xhii + abu_he * (xheii + 2.0 * xheiii) + abu_c)


def thermal(
    dt: float,
    start_temp: float,
    ndens_e: float,
    ndens_a: float,
    heating: float,
    Hz: float,
    xh: None | tuple[float, float, float] = None,
    cool_tables: None | CoolingTables = None,
    relative_denergy: float = 0.1,
    gamma: float = 5.0 / 3.0,
    min_temp: float = 1.0,
    abu_h: float = 0.926,
    abu_he: float = 0.074,
    cosmo_only: bool = False,
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
    xh :
        Tuple of ionized fractions for H and He species (xHI, xHeI, xHeII), required if cosmo_only is False.
    cool_tables :
        Cooling tables for atomic cooling, required if cosmo_only is False.
    relative_denergy :
        Maximum allowed relative change in internal energy per iteration, by default 0.1.
    gamma :
        Adiabatic index of the gas, by default 5/3.
    min_temp :
        Minimum allowed temperature (K), by default 1.0.
    cosmo_only :
        If True, only include cosmological cooling; if False, include both cosmological and atomic
        cooling, and as such xh and cool_tables must be provided. Default False.
    max_iterations :
        Maximum number of iterations to perform, by default 10000.
    """
    if start_temp <= min_temp:
        return start_temp, start_temp

    if not cosmo_only:
        if xh is None:
            raise ValueError("xh must be provided when cosmo_only is False.")
        if cool_tables is None:
            raise ValueError("cool_tables must be provided when cosmo_only is False.")

    ui = get_energy(start_temp, ndens_a + ndens_e, gamma)
    end_temp = start_temp
    avg_temp = 0.0

    tot_time = 0.0
    niter = 0
    while niter < max_iterations and tot_time < dt * (1 - 1e-6):
        rate = heating - cosmo_cooling_rate(ui, Hz)
        if not cosmo_only:
            assert xh is not None and cool_tables is not None
            rate -= cooling_rate(
                ndens_a, ndens_e, end_temp, *xh, cool_tables, abu_h, abu_he
            )
        subdt = min(relative_denergy * ui / abs(rate), dt - tot_time)

        ui += rate * subdt
        avg_temp += 0.5 * end_temp * subdt

        end_temp = get_temperature(ui, ndens_a + ndens_e, gamma)
        avg_temp += 0.5 * end_temp * subdt

        tot_time += subdt
        niter += 1
        if end_temp < min_temp:
            ui = get_energy(min_temp, ndens_a + ndens_e, gamma)
            end_temp = min_temp
            break

    if tot_time > 0:
        avg_temp /= tot_time

    return end_temp, avg_temp
