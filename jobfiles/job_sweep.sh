#!/bin/bash

# Switch to your virtual environment
source /home/gvisona/environments/prot/bin/activate
module load cudnn/8.6.0-cu11.7
module load cuda/11.7


# python3 /home/gvisona/Immunology/hyperparameter_sweeps/Classifier_fromBApretrained_sweep.py  \
# python3 /home/gvisona/Immunology/hyperparameter_sweeps/BetaRegr_fromBApretrained_no_class_sweep.py  \
# python3 /home/gvisona/Immunology/hyperparameter_sweeps/BindingAffinity_sweep.py  \
python3 /home/gvisona/Immunology/hyperparameter_sweeps/BetaRegr_fromBApretrained_sweep.py  \
--seed $1 \

# --experiment_name "test_metrics_sweep" \
# --experiment_group "IEDB_CLIPT5_sweeps" \
# --project_folder "/fast/gvisona/Immunology" \
# --immunogenicity_df "/home/gvisona/Immunology/processed_data/DHLAP_immunogenicity_data.csv" \
# --binding_affinity_df "/home/gvisona/Immunology/processed_data/DHLAP_binding_affinity_data.csv" \
# --pseudo_seq_file "/home/gvisona/Immunology/data/NetMHCpan_pseudoseq/MHC_pseudo.dat" \

# CODE=$?
# if [ $CODE -eq 0 ]
# then
# curl -d "Job N.$1 finished successfully." ntfy.sh/gv_cluster_experiments
# elif [ $CODE -eq 3 ]
# then
# curl -d "Job N.$1 restarted." ntfy.sh/gv_cluster_experiments
# else
# curl -d "Job N.$1 failed with exit code $CODE." ntfy.sh/gv_cluster_experiments
# fi
