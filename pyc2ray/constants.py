"""
Conversion Factors.
When doing direct comparisons with C2Ray, the difference between astropy.constants
and the C2Ray values may be visible, thus we use the same exact value
for the constants. This can be changed to the
astropy values once consistency between the two codes has been established
"""

from typing import Final

from astropy import constants as cst
from astropy import units as u

# Year in seconds
year2s: Final[float] = (1 * u.yr).cgs.value

# eV to Frequency (Hz)
ev2fr: Final[float] = 1.0 / (cst.h * u.Hz).to("eV").value

# eV to Kelvin
ev2k: Final[float] = 1.0 / (cst.k_B * u.K).to("eV").value

# parsec in cm
pc: Final[float] = (1 * u.pc).cgs.value

# kiloparsec in cm
kpc: Final[float] = (1 * u.kpc).cgs.value

# megaparsec in cm
Mpc: Final[float] = (1 * u.Mpc).cgs.value

# solar mass to grams
msun2g: Final[float] = (1 * u.Msun).cgs.value

# proton mass to grams
m_p: Final[float] = cst.m_p.cgs.value
