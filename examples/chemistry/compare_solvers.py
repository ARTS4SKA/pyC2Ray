from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as cst
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

from pyc2ray.load_extensions import libasora, libasora_He


@contextmanager
def device_init():
    libasora.device_init()
    libasora_He.device_init()
    yield
    libasora.device_close()
    libasora_He.device_close()


@contextmanager
def setup_chemistry_doric(
    mesh_size: int = 10, ndens0: float = 1e-7, temp0: float = 1e4
):
    mesh_shape = (mesh_size,) * 3

    # time-step
    dt = (1 * u.Myr).cgs.value

    # density field [g/cm^3]
    ndens = np.full(mesh_shape, ndens0, dtype=np.float64, order="F")

    # temperature [K]
    temp = np.full(mesh_shape, temp0, dtype=np.float64, order="F")

    # Hydrogen ionization fraction
    xh = np.zeros_like(ndens, order="F")
    xh_av = np.zeros_like(ndens, order="F")
    xh_int = np.zeros_like(ndens, order="F")

    # photo-ionization rate [s^-1]
    phi_ion = np.full_like(ndens, 1e-14)

    # clumping factor
    clump = np.ones_like(ndens)

    # constants
    eth0 = 13.598
    bh00 = 2.59e-13
    colh0 = 1.079e-8 / eth0**2
    albpow = -0.7
    temph0 = eth0 / (cst.k_B * u.K).to("eV").value
    abu_c = 7.1e-7

    yield (
        dt,
        ndens,
        temp,
        xh,
        xh_av,
        xh_int,
        phi_ion,
        clump,
        bh00,
        albpow,
        colh0,
        temph0,
        abu_c,
    )


@contextmanager
def setup_chemistry_friedrich(
    mesh_size: int = 10,
    ndens0: float = 1e-7,
    temp0: float = 1e4,
    ionize_species: tuple[bool, bool, bool] = (True, True, True),
):
    assert len(ionize_species) == 3, "ionize_species should be a tuple of 3 booleans"

    mesh_shape = (mesh_size,) * 3

    dt = (1 * u.Myr).cgs.value
    Hz = cosmo.H(10).cgs.value

    # density field [g/cm^3]
    ndens = np.full(mesh_shape, ndens0, dtype=np.float64, order="F")

    # temperature [K]
    temp = np.full(mesh_shape, temp0, dtype=np.float64, order="F")

    # Ionization fractions for x, x_av and x_int
    xHIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    xHeIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    xHeIIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    phion = tuple(
        np.full_like(ndens, s * p)
        for s, p in zip(ionize_species, (1e-14, 1e-15, 1e-16))
    )
    # Unused right now
    pheat = tuple(np.zeros_like(ndens) for _ in ionize_species)

    # Clumping factor
    clump = np.ones_like(ndens)

    yield (
        dt,
        Hz,
        ndens,
        temp,
        temp,
        *xHIIs,
        *xHeIIs,
        *xHeIIIs,
        *phion,
        *pheat,
        clump,
        32,
    )


def compare_chemistry(iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    kwargs = dict(mesh_size=1, ndens0=1e-7, temp0=1e4)
    d_xh: list[float] = []
    f_xh: list[float] = []
    with (
        device_init(),
        setup_chemistry_doric(**kwargs) as dargs,
        setup_chemistry_friedrich(**kwargs) as fargs,
    ):
        d_xHII, d_xHII_av, d_xHII_int = dargs[3:6]
        f_xHII, f_xHII_av, f_xHII_int = fargs[5:8]

        # Not physically meaningful, just to test the chemistry solvers
        for _ in range(iterations):
            libasora.chemistry_global_pass(*dargs)
            libasora_He.chemistry_global_pass(*fargs)

            # Compare results
            d_xh.append((d_xHII_int.item(), d_xHII_av.item()))
            f_xh.append((f_xHII_int.item(), f_xHII_av.item()))

            # Update states with same values to keep initial conditions in sync
            d_xHII[:] = d_xHII_int
            f_xHII[:] = d_xHII_int

    return np.array(d_xh), np.array(f_xh)


if __name__ == "__main__":
    d_xh, f_xh = compare_chemistry()

    xh_diff = abs(d_xh[:, 0] - f_xh[:, 0])
    xh_av_diff = abs(d_xh[:, 1] - f_xh[:, 1])

    plt.plot(xh_diff, label="xHII")
    plt.plot(xh_av_diff, label="xHII_av")

    plt.xticks(range(0, len(xh_diff), 2))
    plt.xlabel("Iteration")
    plt.ylabel("Absolute difference")

    plt.legend()
    plt.tight_layout()

    # Differences between the two solvers are due to the thermal evolution
    # and presence of Helium (abu_he parameter) in Friedrich
    plt.savefig(Path(__file__).parent / "chemistry_comparison.png")
    plt.show()
