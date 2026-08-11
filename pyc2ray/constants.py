"""Conversion factors and constants defined in CGS units."""

import math
from typing import Final

from astropy import constants as cst
from astropy import units as u

# Year in seconds
year2s: Final[float] = (1.0 * u.yr).cgs.value

# eV to Frequency (Hz)
ev2hz: Final[float] = (1.0 * u.eV / cst.h).to("Hz").value

# eV to Kelvin
ev2k: Final[float] = (1.0 * u.eV / cst.k_B).to("K").value

# parsec in cm
pc: Final[float] = (1.0 * u.pc).cgs.value

# kiloparsec in cm
kpc: Final[float] = (1.0 * u.kpc).cgs.value

# megaparsec in cm
Mpc: Final[float] = (1.0 * u.Mpc).cgs.value

# solar mass to grams
M_sun: Final[float] = cst.M_sun.cgs.value

# solar radius to grams
R_sun: Final[float] = cst.R_sun.cgs.value

# proton mass to grams
m_p: Final[float] = cst.m_p.cgs.value

# Boltzmann constant in erg/K
k_B: Final[float] = cst.k_B.cgs.value

# Plank constant in erg * s
h: Final[float] = cst.h.cgs.value

# Speed of light cm / s
c: Final[float] = cst.c.cgs.value

# pi
pi: Final[float] = math.pi
