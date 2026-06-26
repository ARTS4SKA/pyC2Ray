#!/bin/bash
set -e

python -m venv env
source env/bin/activate

python -m pip install .[test]
pytest

deactivate
