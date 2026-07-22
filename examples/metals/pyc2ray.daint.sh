#!/bin/sh
#SBATCH --job-name=pyc2ray	# flag name (your choice)
#SBATCH --account=c45		# account of the computing allocation
#SBATCH --nodes=1		# number of nodes to be used
#SBATCH --ntasks-per-node=4	# number of MPI task per node
#SBATCH --constraint=gpu	# switch one GPU usage
#SBATCH --gres=gpu:4		# number of GPU per node (Daint has 4 gpus per node)
##SBATCH --time=12:00:00		# length of the run (max 24 hours)
#SBATCH -p debug --time=00:30:00	# comment this in if you want to run in debug mode (max 30min)
#SBATCH -e logs/pyc2ray.%j.err		# some printout of your run for errors
#SBATCH -o logs/pyc2ray.%j.out		# some printout of your run
#SBATCH --mail-type=ALL			# to get email on the run
#SBATCH --mail-user=cprior@ethz.ch

## SBATCH --uenv=prgenv-gnu/24.11:v2 --view=default	# activate the user enviromnet (similar to Jupyter Lab)
#SBATCH --uenv-passthrough=use 

export LD_LIBRARY_PATH=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/lib-abi-mpich:$LD_LIBRARY_PATH
export HDF5_USE_FILE_LOCKING='FALSE'
export MPICH_GPU_SUPPORT_ENABLED=0

# activate env
source $HOME/venvs/pyc2ray-env/bin/activate

# for some reason in SLURM with uenv, it does fetch the wrong python. This takes the correct one.
export PYTHONBIN=$(which python)
echo $PYTHONBIN

# run code
srun $PYTHONBIN run_test.py parameters.yml 
deactivate
