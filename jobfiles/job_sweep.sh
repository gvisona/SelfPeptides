#!/bin/bash

# Switch to your virtual environment
source /home/gvisona/environments/prot/bin/activate
module load cudnn/8.6.0-cu11.7
module load cuda/11.7



# python3 /home/gvisona/SelfPeptides/hyperparameter_sweeps/BindingAffinity_sweep.py  \
# python3 /home/gvisona/SelfPeptides/hyperparameter_sweeps/Classifier_sweep.py  \
python3 /home/gvisona/SelfPeptides/hyperparameter_sweeps/Immunogenicity_sweep.py  \
--seed $1 \
