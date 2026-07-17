import logging
import sys
sys.path.append("../../../")
import pyc2ray as pc2r
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
parser.add_argument("--plot",action='store_true')
args = parser.parse_args()

# Global parameters
num_steps_between_slices = 4
numzred = 4
paramfile = "parameters.yml"
avgdens = 1.0e-6

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile)

# Get logger
logger = logging.getLogger("pyc2ray.c2ray_test")

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,1e7)

# Read sources
numsrc = 100
srcpos, srcflux = sim.read_sources("src_mult_100.txt",numsrc)

# Measure time
tinit = time.time()

for k in range(len(zred_array)-1):

    # Compute timestep of current redshift slice
    zi = zred_array[k]
    zf = zred_array[k+1]
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)

    # Write output
    sim.write_output(zi)

    # Set density field (could be an actual cosmological field here)
    # TODO: this has to set the comoving density which is then scaled to the
    # correct redshift. In the timesteps, the density is then "diluted" gradually
    sim.set_constant_average_density(avgdens,0) 

    logger.info("=================================")
    logger.info(f"Doing redshift {zi:.3f} to {zf:.3f}")
    logger.info("=================================")

    # Do num_steps_between_slices timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        logger.info(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n")

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing (checked internally by the class)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcflux, srcpos)

if args.plot:
    import matplotlib.pyplot as plt
    plt.imshow(sim.xh[:,:,64],norm='log',cmap='jet')
    #plt.imshow(sim.phi_ion[:,:,64],norm='log',cmap='inferno')
    plt.colorbar()
    plt.show()

# Write final output
sim.write_output(zf)
logger.info(f"Done. Final time: {time.time() - tinit : .3f} seconds")
