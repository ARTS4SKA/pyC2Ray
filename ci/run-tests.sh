#!/bin/bash
set -e

python -m venv env
source env/bin/activate

# Install mpi4py from source to ensure it is built against the correct MPI library
MPICC=mpicc python -m pip install --no-binary=mpi4py --no-cache-dir mpi4py
python -m pip install .[test]

pytest

deactivate
