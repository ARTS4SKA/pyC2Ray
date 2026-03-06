import sys
import numpy as np
import shutil
import time
import pyc2ray as pc2r
import logging
import os

PathType = str | os.PathLike

# ======================================================================
# Example for pyc2ray: Cosmological simulation from N-body
# ======================================================================
logger = logging.getLogger("pyc2ray")


def run_simulation(paramfile: PathType, num_steps_between_slices: int = 2) -> None:
    """
    Parameter
    ----------
    paramfile :
        Name of a YAML file containing parameters for the C2Ray simulation
    num_steps_between_slices :
        Number of timesteps between redshift slices (default: 2)
    """
    # Create C2Ray object
    sim = pc2r.C2Ray_fstar(paramfile=paramfile)

    # Copy parameter file into the output directory
    if sim.rank == 0:
        shutil.copy(paramfile, sim.results_basename)

    # Get redshift list (test case)
    idx_zred, zred_array = np.loadtxt(
        sim.inputs_basename / "redshift_checkpoints.txt", unpack=True
    )

    # check for resume simulation
    if sim.resume:
        i_start = (zred_array > sim.zred).nonzero()[0][-1]
        sim.resume = i_start + 1
    else:
        i_start = 0

    # Measure time
    timer = pc2r.Timer()
    timer.start()

    # Loop over redshifts
    for k in range(i_start, len(zred_array) - 1):
        iz = idx_zred[k]  # Index redshift
        zi = zred_array[k]  # Start redshift
        zf = zred_array[k + 1]  # End redshift

        logger.info(
            "\n=================================\n"
            f"Doing redshift {zi:.3f} to {zf:.3f}"
            "\n=================================\n"
        )

        # Compute timestep of current redshift slice
        dt = sim.set_timestep(zi, zf, num_steps_between_slices)

        # Read input files
        # FIXME: This should come from parameter file
        sim.read_density(f"CDM_100Mpc_2048.{iz:05d}.ovrden.npy", z=zi)

        # Read source files
        # FIXME: This should come from parameter file
        srcpos, normflux = sim.ionizing_flux(
            f"CDM_100Mpc_2048.{iz:05d}.halo.txt", z=zi, dt=dt
        )

        # save previous time-step output (or initial state)
        if sim.rank == 0 and k != i_start:
            sim.write_output(z=zi, ext=".npy")

        # Set redshift to current slice redshift
        sim.zred = zi

        # Loop over timesteps
        for t in range(num_steps_between_slices):
            # get cosmological time of the intermediate time-steps
            t_age = sim.cosmology.age(zi).cgs.value + t * dt

            # get corresponding redshift
            z = sim.time2zred(t_age)

            # register wall clock time
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

    # stop the timer and print the summary
    timer.stop()
    sim.printlog(timer.summary, sim.logfile)

    # Write final output
    sim.write_output(zf, ext=".npy")


if __name__ == "__main__":
    paramfile = sys.argv[1]
    sys.exit(run_simulation(paramfile))
