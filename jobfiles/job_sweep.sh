#!/bin/bash

# Switch to your virtual environment
source /home/gvisona/environments/prot/bin/activate
module load cudnn/8.6.0-cu11.7
module load cuda/11.7



python3 /home/gvisona/Immunology/hyperparameter_sweeps/Classifier_sweep.py  \
--seed $1 \
