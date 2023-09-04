#!/bin/bash

# Switch to your virtual environment
source /home/gvisona/environments/prot/bin/activate
module load cudnn/8.6.0-cu11.7
module load cuda/11.7


learning_rates=(0.00014 0.000028 0.000014 0.0000028 0.0000014)
lr=${learning_rates[@]:${idx}:1}


idx=$1
sd=$idx
lr=1.0
wd=0.0001

python3 /home/gvisona/SelfPeptides/training_scripts/train_binding_model.py  \
--seed "$sd" --run_number "$idx" \
--accumulate_batches "64" --lr  "$lr" \
--weight_decay "$wd" \
--val_size 0.1 --test_size 0.15 \
--validate_every_n_updates 32 \
--max_updates 5000 --dropout_p 0.05 \
--ramp_up 0.1 --min_frac 0.1 \
--momentum 0.9 --nesterov_momentum \
--embedding_dim 512 --mlp_hidden_dim 2048 --mlp_num_layers 2 --output_dim 1 \
--PMA_num_heads 4 --PMA_ln --num_heads 4 --transf_hidden_dim 512 --n_attention_layers 2 \
--batch_size 32 --num_workers 1 \
--hla_repr "Allele Pseudo-sequence" --early_stopping --patience 200\
--experiment_group "Binding_model_masking" --experiment_name "BA_tune_devel" \
--project_folder "/fast/gvisona/SelfPeptides" \
--pretrained_aa_embeddings "/home/gvisona/SelfPeptides/processed_data/aa_embeddings/learned_BA_AA_embeddings.npy" \
--binding_affinity_df "/home/gvisona/SelfPeptides/processed_data/Binding_Affinity/DHLAP_binding_affinity_data.csv" \
--pseudo_seq_file "/home/gvisona/SelfPeptides/data/NetMHCpan_pseudoseq/MHC_pseudo.dat" 
# --cool_down 0.6
# --init_checkpoint "/home/gvisona/SelfPeptides/trained_models/binding_model_OLD/checkpoint.pt" \