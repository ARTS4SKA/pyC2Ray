import argparse
import logging
import os
import shutil
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

import pyc2ray as pc2r
from pyc2ray.utils.other_utils import find_redshit_index

PathType = str | os.PathLike

logger = logging.getLogger("pyc2ray")


def main(
    parameters: PathType,
    sub_time_steps: int = 2,
    z_start: float | None = None,
    z_end: float | None = None,
) -> int:
    """
    Parameter
    ----------
    parameters :
        Name of a YAML file containing parameters for the C2Ray simulation
    sub_time_steps :
        Number of timesteps between redshift slices (default: 2)
    """
    # Create C2Ray object
    sim = pc2r.C2Ray_fstar(paramfile=parameters)

    # Copy parameter file into the output directory
    if sim.rank == 0:
        shutil.copy(parameters, sim.results_basename)

    # Get list of redshift slices to simulate
    zred_idx, zred_array = np.loadtxt(
        sim.inputs_basename / "redshift_checkpoints.txt", unpack=True
    )

    start_idx = 0
    if sim.resume or z_start is not None:
        z_start = min(sim.zred, z_start) if z_start is not None else sim.zred
        start_idx = find_redshit_index(zred_array, z_start)

    end_idx = len(zred_array)
    if z_end is not None:
        end_idx = find_redshit_index(zred_array, z_end)

    # Measure time
    timer = pc2r.Timer()
    timer.start()

    # Loop over redshifts
    for k, (zi, zf) in enumerate(pairwise(zred_array[start_idx:end_idx]), start_idx):
        iz = int(zred_idx[k])
        logger.info(
            "\n=================================\n"
            f"Doing redshift {zi:.3f} to {zf:.3f}"
            "\n=================================\n"
        )

        # Compute timestep of current redshift slice
        dt = sim.set_timestep(zi, zf, sub_time_steps)

        # Read input files
        # FIXME: This should come from parameter file
        sim.read_density(f"CDM_100Mpc_2048.{iz:05d}.ovrden.npy", z=zi)

        # Read source files
        # FIXME: This should come from parameter file
        srcpos, normflux = sim.ionizing_flux(
            f"CDM_100Mpc_2048.{iz:05d}.halo.txt", z=zi, dt=dt
        )

        # Save previous time-step output (or initial state)
        if sim.rank == 0 and k != start_idx:
            sim.write_output(z=zi, ext=".npy")

        # Set redshift to current slice redshift
        sim.zred = zi

        # Loop over timesteps
        for t in range(sub_time_steps):
            # Get cosmological time of the intermediate time-steps
            t_age = sim.cosmology.age(zi).cgs.value + t * dt

            # Get corresponding redshift
            z = sim.time2zred(t_age)

            # Register wall clock time
            tnow = timer.lap(f"z = {z:.3f}")
            logger.info(
                f"\n --- Timestep {t + 1}: z = {sim.zred:.3f}, Wall clock time: {tnow} --- \n"
            )

            # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
            sim.cosmo_evolve(dt)

            # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
            sim.evolve3D(dt, normflux, srcpos)

        # Evolve cosmology over final half time step to reach the correct time for next slice (see note in c2ray_base.py)
        sim.cosmo_evolve_to_now()

    # Write final output
    sim.write_output(zf, ext=".npy")

    # stop the timer and print the summary
    timer.stop()
    logger.info(timer.summary)

    return 0


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run an F* pyC2Ray simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("parameters", type=Path, help="Path to parameter file")
    parser.add_argument(
        "-t",
        "--sub-time-steps",
        type=int,
        default=2,
        help="Number of time steps between redshift slices",
    )
    parser.add_argument("--z-start", type=float, help="Starting redshift (optional)")
    parser.add_argument("--z-end", type=float, help="Ending redshift (optional)")

    return vars(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main(**parse_args()))
