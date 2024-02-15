#!/bin/bash

# Switch to your virtual environment
source /home/gvisona/environments/prot/bin/activate
module load cudnn/8.6.0-cu11.7
module load cuda/11.7


# learning_rates=(0.00014 0.000028 0.000014 0.0000028 0.0000014)
# lr=${learning_rates[@]:${idx}:1}


# idx=$1
# sd=90412
# wd=0.0001

python3 /home/gvisona/SelfPeptides/training_scripts/train_immunogenicity_model.py  \
--seed 42 --run_number 0 \
--experiment_group "BetaRegression_MeanRegression" --experiment_name "BetaRegr_Exploration" \
--kl_loss_type "forward" \
--accumulate_batches 4 --batch_size 32 \
--weight_decay 0.0001 --lr  0.0003 \
--min_subjects_tested 1 \
--val_size 0.1 --test_size 0.15 \
--validate_every_n_updates 32 \
--max_updates 10000 --dropout_p 0.05 \
--ramp_up 0.1 --min_frac 0.1 \
--IR_n_bins 50 --LDS_kernel "triang" --LDS_ks 11 --LDS_sigma 1.0 \
--loss_weights "LDS_weights" --constraints_weight 0.01 \
--imm_regression_hidden_dim 2048 \
--mlp_input_dim 1025 --mlp_output_dim 512 \
--mlp_hidden_dim 2048 --mlp_num_layers 2 \
--early_stopping --patience 2000 \
--project_folder "/fast/gvisona/SelfPeptides" \
--init_checkpoint "/home/gvisona/SelfPeptides/trained_models/binding_model_OLD/checkpoint.pt" \
--pseudo_seq_file "/home/gvisona/SelfPeptides/processed_data/HLA_embeddings/HLA_pseudoseqs_T5/hla_pseudoseq_mapping.csv" \
--hla_prot_seq_file "/home/gvisona/SelfPeptides/processed_data/HLA_embeddings/HLA_proteins_T5/hla_proteins_mapping.csv" \
--immunogenicity_df "/home/gvisona/SelfPeptides/processed_data/Immunogenicity/Processed_TCell_IEDB_beta_summed.csv" \
--dhlap_df "/home/gvisona/SelfPeptides/processed_data/Immunogenicity/DHLAP_immunogenicity_data.csv" \
--binding_model_config "/home/gvisona/SelfPeptides/trained_models/BindingModels/floral-sweep-3/config.json" \
--binding_model_checkpoint "/home/gvisona/SelfPeptides/trained_models/BindingModels/floral-sweep-3/checkpoints/001_checkpoint.pt" \
--force_restart