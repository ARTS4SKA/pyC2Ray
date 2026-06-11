import argparse
import logging
import os
import sys
from pathlib import Path

import pyc2ray as pc2r

PathType = str | os.PathLike

logger = logging.getLogger("pyc2ray")


def main(
    paramfile: PathType,
    source_file: Path,
    numzred: int,
    t_evol: float = 5e8,
    num_steps_between_slices: int = 1,
) -> None:
    # Create C2Ray object
    sim = pc2r.C2Ray_Test(paramfile)

    # Generate redshift list (test case)
    zreds = sim.generate_redshift_array(numzred + 1, t_evol / numzred)

    # Read sources
    srcpos, srcstrength = sim.read_sources(source_file, 1)

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
        sim.density_init(zi)

        # Set redshift to current slice redshift
        sim.zred = zi

        # Loop over timesteps
        for t in range(num_steps_between_slices):
            # Get cosmological time of the intermediate time-steps
            t_age = sim.cosmology.age(zi).cgs.value + t * dt
            z = sim.time2zred(t_age)

            tnow = timer.lap(f"z = {z:.3f}")
            logger.info(
                f"--- Timestep {t + 1:n}. Redshift: z = {sim.zred: .3f} Wall clock time: {tnow} seconds ---"
            )

            # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
            # If cosmology is disabled in parameter, this step does nothing
            sim.cosmo_evolve(dt)

            # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
            sim.evolve3D(dt, srcstrength, srcpos)

    # Write final output
    sim.write_output(zf)

    timer.stop()
    logger.info(timer.summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", type=Path, help="Path to parameter file")
    source_default = Path(__file__).parent / "source.txt"
    source_default_rel = source_default.relative_to(Path.cwd(), walk_up=True)
    parser.add_argument(
        "-s",
        "--source-file",
        type=Path,
        default=source_default,
        help=f"Path to source file (default: {source_default_rel})",
    )
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
    args = parser.parse_args()

    sys.exit(main(args.paramfile, args.source_file, args.numzred, args.tevolve))
