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
    avgdens: float = 1.0e-6,
    delta_t: float = 1e7,
    num_src: int = 5,
    num_steps_between_slices: int = 10,
    show_plot: bool = False,
) -> None:
    # Create C2Ray object
    sim = pc2r.C2Ray_Test(paramfile)

    # Generate redshift list (test case)
    zreds = sim.generate_redshift_array(numzred, delta_t)

    # Read sources
    srcpos, srcflux = sim.read_sources(source_file, num_src)

    # Measure time
    timer = pc2r.Timer()
    timer.start()

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

        # Set density field (could be an actual cosmological field here)
        # TODO: this has to set the comoving density which is then scaled to the
        # correct redshift. In the timesteps, the density is then "diluted" gradually
        sim.set_constant_average_density(avgdens, 0)

        # Set redshift to current slice redshift
        sim.zred = zi

        # Do num_steps_between_slices timesteps
        for t in range(num_steps_between_slices):
            # Get cosmological time of the intermediate time-steps
            t_age = sim.cosmology.age(zi).cgs.value + t * dt
            z = sim.time2zred(t_age)
            tnow = timer.lap(f"z = {z:.3f}")
            logger.info(
                f"\n --- Timestep {t + 1:n}. Redshift: z = {sim.zred: .3f} Wall clock time: {tnow} seconds --- \n"
            )

            # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
            # If cosmology is disabled in parameter, this step does nothing (checked internally by the class)
            sim.cosmo_evolve(dt)

            # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
            sim.evolve3D(dt, srcflux, srcpos)

    # Write final output
    sim.write_output(zf)

    timer.stop()
    logger.info(timer.summary)

    if show_plot:
        import matplotlib.pyplot as plt

        plt.imshow(sim.xh[:, :, 64], norm="log", cmap="jet")
        # plt.imshow(sim.phi_ion[:,:,64],norm='log',cmap='inferno')
        plt.colorbar()
        plt.show()


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
        default=2,
        help="Number of redshift slices to evolve",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show plot of final ionization state"
    )
    args = parser.parse_args()

    sys.exit(
        main(
            args.paramfile,
            args.source_file,
            args.numzred,
            show_plot=args.plot,
        )
    )
