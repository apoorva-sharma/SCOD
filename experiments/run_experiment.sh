#!/bin/bash
set -e

# python train_models.py $1
# python create_unc_models.py $1
python test_unc_models.py $1
python create_figures.py $1