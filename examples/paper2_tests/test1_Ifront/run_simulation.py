import argparse
import logging
import os
import sys
import shutil
from pathlib import Path

import numpy as np

import pyc2ray as pc2r

PathType = str | os.PathLike

logger = logging.getLogger("pyc2ray")


def main(
    paramfile: PathType,
    numzred: int,
    t_evol: float = 5e8,
    flux_strength: float = 1e54,
    avg_dens: float = 1.87e-7,
    num_steps_between_slices: int = 2,
) -> int:
    # Create C2Ray object
    sim = pc2r.C2Ray_Test(paramfile)
    shutil.copy(paramfile, sim.results_basename / paramfile.name)

    # Generate redshift list (test case)
    zreds = sim.generate_redshift_array(numzred + 1, t_evol / numzred)

    # Read sources
    # Default source position for test case
    src_pos = np.expand_dims(sim.shape, 1) // 2
    src_flux = np.array([flux_strength]) / 1e48
    logger.info(
        f"Placing a single source at the center of the box ({src_pos.ravel()}) with flux {flux_strength}"
    )

    # Measure time
    timer = pc2r.Timer()
    timer.start()

    # Loop over redshifts
    for zi, zf in zip(zreds[:-1], zreds[1:]):
        logger.info(
            "\n=================================\n"
            f"Doing redshift {zi:.3f} to {zf:.3f}"
            "\n=================================\n"
        )

        # Compute timestep of current redshift slice
        dt = sim.set_timestep(zi, zf, num_steps_between_slices)

        # Write output
        sim.write_output(zi)

        # Set density (when cosmological is false, zi has no effect)
        sim.set_constant_average_density(avg_dens, zi)

        # Set redshift to current slice redshift
        sim.zred = zi

        # Loop over timesteps
        for t in range(num_steps_between_slices):
            # Get cosmological time of the intermediate time-steps
            t_age = sim.cosmology.age(zi).cgs.value + t * dt
            z = sim.time2zred(t_age)

            tnow = timer.lap(f"z = {z:.3f}")
            logger.info(
                f"--- Timestep {t + 1:n}. Redshift: z = {sim.zred:.3f} Wall clock time: {tnow} seconds ---"
            )

            # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
            # If cosmology is disabled in parameter, this step does nothing
            sim.cosmo_evolve(dt)

            # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
            sim.evolve3D(dt, src_flux, src_pos)

        sim.cosmo_evolve_to_now()

    # Write final output
    sim.write_output(zf)

    timer.stop()
    logger.info(timer.summary)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", type=Path, help="Path to parameter file")
    parser.add_argument(
        "-z",
        "--numzred",
        type=int,
        default=10,
        help="Number of redshift slices to evolve",
    )
    parser.add_argument(
        "-t",
        "--tevolve",
        type=float,
        default=5e8,
        help="Total evolution time in years (default: 5e8)",
    )
    parser.add_argument(
        "-f",
        "--flux",
        type=float,
        default=1e54,
        help="Flux strength of the single source (default: 1e54)",
    )
    parser.add_argument(
        "-n",
        "--avg-density",
        type=float,
        default=1.87e-7,
        help="Average number density in cm^-3 (default: 1.87e-7)",
    )
    args = parser.parse_args()

    sys.exit(
        main(args.parameters, args.numzred, args.tevolve, args.flux, args.avg_density)
    )
