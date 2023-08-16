#!/bin/bash

# Switch to your virtual environment
source /home/gvisona/environments/prot/bin/activate
module load cudnn/8.6.0-cu11.7
module load cuda/11.7

idx=$1+1

learning_rates=(0.00003 0.00048 0.00192 0.00003 0.00048 0.00192)
weight_decays=(0.000001 0.000001 0.000001 0.00001 0.00001 0.00001)
acc_batches=(1 16 64 1 16 64)

lr=${learning_rates[idx]}
wd=${weight_decays[idx]}
ab=${acc_batches[idx]}


python3 /home/gvisona/SelfPeptides/training_scripts/train_selfpeptide_embedder_cmt.py  \
--seed $1 \
--accumulate_batches $ab --lr  $lr \
--weight_decay $wd \
--val_size 10000 --test_size 10000 --ref_size 10000 \
--validate_every_n_updates 100000 \
--max_updates 20000000 --dropout_p 0.0 \ 
--ramp_up 0.3 --min_frac 0.1 --cool_down 0.6  \
--momentum 0.9 --nesterov_momentum \
--embedding_dim 512 --projection_hidden_dim 2048 --projection_dim 32 \
--PMA_num_heads 1 --PMA_ln --num_heads 4 --transf_hidden_dim 2048 --n_attention_layers 2 \
--batch_size 32 \
--experiment_group "SP_Embedder" --experiment_name "Embeddings_CMT_long" \
--project_folder "/fast/gvisona/SelfPeptides" \
--hdf5_dataset "/home/gvisona/SelfPeptides/processed_data/Self_nonSelf/pre_tokenized_peptides_dataset.hdf5" \
--pretrained_aa_embeddings "/home/gvisona/SelfPeptides/processed_data/aa_embeddings/learned_BA_AA_embeddings.npy" \
--reg_weight 0.0001 --margin 0.6 --loss_s 10.0 