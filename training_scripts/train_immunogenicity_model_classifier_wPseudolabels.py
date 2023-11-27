
import pandas as pd
import numpy as np
import argparse
from argparse import ArgumentParser 
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from os.path import exists, join
from tqdm import tqdm
from selfpeptide.utils.data_utils import load_immunogenicity_dataframes_calibration, load_binding_affinity_dataframes, load_immunogenicity_dataframes
from selfpeptide.utils.data_utils import SequencesInteractionDataset_returnHLA, SequencesInteractionDataset
from selfpeptide.utils.training_utils import get_class_weights, sigmoid
from selfpeptide.utils.training_utils import lr_schedule, warmup_constant_lr_schedule, eval_classification_metrics
from selfpeptide.model.immunogenicity_predictor import JointPeptidesNetwork_Classifier
from selfpeptide.utils.constants import *

def train(config=None, init_wandb=True):
    # start a new wandb run to track this script
    if init_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=config['experiment_name'],
            
            # track hyperparameters and run metadata
            config=config
        )
    
        config = wandb.config
    
    run_name = wandb.run.name
    if config.get("run_number", None) is None:
        run_number = config["seed"]
    else:
        run_number = config["run_number"]
        
    if run_name is None or len(run_name) < 1 or not config["wandb_sweep"]:
        run_name = str(run_number)

    output_folder = join(config['project_folder'],
                         "outputs",
                         config['experiment_group'],
                         config['experiment_name'],
                         run_name)
    os.makedirs(output_folder, exist_ok=True)

    results_folder = join(config['project_folder'],
                         "outputs",
                         config['experiment_group'],
                         config['experiment_name'],
                         "results")
    os.makedirs(results_folder, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Loading immunogenicity data")
    imm_df, dhlap_imm_df = load_immunogenicity_dataframes(config, split_data=False)
    train_df = pd.read_csv(join(config["data_folder"], "train_set.csv"))
    val_df = pd.read_csv(join(config["data_folder"], "val_set.csv"))
    test_df = pd.read_csv(join(config["data_folder"], "test_set.csv"))
    
    gen_df = pd.read_csv(join(config["data_folder"], "generated_pseudolabels_set.csv"))
    
    if config.get("gen_negative_samples", False):
        ba_df = load_binding_affinity_dataframes(config, split_data=False)
        ba_df = ba_df[ba_df["Label"]==0]
        ba_df["Peptide Length"] = ba_df["Peptide"].str.len()
        ba_df = ba_df[(ba_df["Peptide Length"]>=MIN_PEPTIDE_LEN)&(ba_df["Peptide Length"]<=MAX_PEPTIDE_LEN)]
        
        iedb_samples = set(tuple(x) for x in train_df[["Peptide", "HLA"]].values).union(
            set(tuple(x) for x in val_df[["Peptide", "HLA"]].values)).union(
            # set(tuple(x) for x in calib_df[["Peptide", "HLA"]].values)).union(
            set(tuple(x) for x in test_df[["Peptide", "HLA"]].values))
        ba_samples = set(tuple(x) for x in ba_df[["Peptide", "HLA"]].values)
        ba_df = ba_df[ba_df[["Peptide", "HLA"]].apply(tuple, 1).isin(ba_samples.difference(iedb_samples))]
        ba_df["Target"] = 0
        gen_df = pd.concat([gen_df, ba_df[["Peptide", "HLA", "Allele Pseudo-sequence", "Target"]]])
        
    
    train_df = pd.concat([train_df, gen_df])
    train_df = train_df.sample(frac=1, replace=False)
        
        
    # train_df.to_csv(join(output_folder, "train_set.csv"), index=False)
    # val_df.to_csv(join(output_folder, "val_set.csv"), index=False)
    # # calib_df.to_csv(join(output_folder, "calib_set.csv"), index=False)
    # test_df.to_csv(join(output_folder, "test_set.csv"), index=False)

    with open(join(output_folder, "config.json"), "w") as f:
        json.dump(dict(config), f)
        
    checkpoint_path = os.path.join(output_folder, "checkpoint.pt")
    
    pos_weight, neg_weight = get_class_weights(train_df, target_label="Target")
    
    wandb.run.summary["pos_weight"] = pos_weight
    wandb.run.summary["neg_weight"] = neg_weight
    
    target_metric = config.get("target_metric", "AUPRC")

    
    train_dset = SequencesInteractionDataset(train_df, hla_repr=config["hla_repr"], target_label="Target")
    val_dset = SequencesInteractionDataset(val_df, hla_repr=config["hla_repr"], target_label="Target")
    # calib_dset = SequencesInteractionDataset(calib_df, hla_repr=config["hla_repr"], target_label="Target")
    test_dset = SequencesInteractionDataset(test_df, hla_repr=config["hla_repr"], target_label="Target")
    
    train_loader = DataLoader(train_dset, batch_size=config['batch_size'], drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=config['batch_size'], drop_last=False, shuffle=False)
    # calib_loader = DataLoader(calib_dset, batch_size=config['batch_size'], drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=config['batch_size'], drop_last=False)
    
    print("Building model")
    model = JointPeptidesNetwork_Classifier(config, config["binding_model_config"], config["sns_model_config"], 
                             binding_checkpoint=config["binding_model_checkpoint"], 
                             sns_checkpoint=config["sns_model_checkpoint"],
                             device=device)
    
    
    if config.get("resume_checkpoint_directory", None) is not None:
        print("Resuming training from model ", config["resume_checkpoint_directory"])
        resume_checkpoint_path = os.path.join(config["resume_checkpoint_directory"], "checkpoint.pt")
        model.load_state_dict(torch.load(resume_checkpoint_path, map_location=device))    
        
    model.to(device)
    
    for p in model.parameters():
        if not p.requires_grad:
            continue
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    
    with open(join(output_folder, "architecture.txt"), "w") as f:
        f.write(model.__repr__())
        
    max_iters = config["max_updates"] * config["accumulate_batches"] + 1
    
    if config["use_class_weights"]:
        class_weights = [neg_weight, pos_weight]
        bce_pos_weight = pos_weight/neg_weight
    else:
        class_weights = [1.0, 1.0]
        bce_pos_weight = 1
        
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight], device=device))
    
    checkpoints_folder = os.path.join(output_folder, "checkpoints")
    os.makedirs(checkpoints_folder, exist_ok=True)
    existing_checkpoints = os.listdir(checkpoints_folder)
    existing_checkpoints = sorted([fname for fname in existing_checkpoints if "_checkpoint.pt" in fname])
    
    resume_checkpoint_path = None
    resume_config = None
    if len(existing_checkpoints)<1:
        checkpoint_fname = "001_checkpoint.pt"
    else:
        last_checkpoint_fname = existing_checkpoints[-1]
        resume_checkpoint_label = last_checkpoint_fname.split(".")[0]
        with open(os.path.join(checkpoints_folder, resume_checkpoint_label+".json"), "r") as f:
            resume_config = json.load(f)
        
        resume_checkpoint_path = os.path.join(checkpoints_folder, last_checkpoint_fname)
        checkpoint_number = int(last_checkpoint_fname.split("_")[0]) + 1
        checkpoint_number = str(checkpoint_number).zfill(3)
        checkpoint_fname = checkpoint_number + "_checkpoint.pt"
        
    if config.get("init_checkpoint", None) is not None and resume_checkpoint_path is None:
        print("Initializing model using checkpoint {}".format(config["init_checkpoint"]))
        model.load_state_dict(torch.load(config["init_checkpoint"]))
        wandb.run.summary["checkpoints/Init_checkpoint"] = config["init_checkpoint"]
    
    
    if resume_checkpoint_path is not None and not config["force_restart"]:
        print("Resuming training from checkpoint {}".format(resume_checkpoint_path))
        model.load_state_dict(torch.load(resume_checkpoint_path))
        wandb.run.summary["checkpoints/Resuming_checkpoint"] = resume_checkpoint_path
        
        
    checkpoint_path = os.path.join(checkpoints_folder, checkpoint_fname)
    wandb.run.summary["checkpoints/Checkpoint_path"] = checkpoint_path
    checkpoint_label = checkpoint_fname.split(".")[0]
    
    
    optimizer = torch.optim.SGD(model.immunogenicity_model.parameters(), lr=config['lr'], momentum=config.get("momentum", 0.9),
                                nesterov=config.get("nesterov_momentum", False),
                                weight_decay=config['weight_decay'])
    
    if resume_checkpoint_path is None and config.get("init_checkpoint", None) is None:
        def lr_lambda(s): return warmup_constant_lr_schedule(s, min_frac=config['min_frac'], total_iters=config["max_updates"],
                                         ramp_up=config['ramp_up'])
    else:
        lr_lambda = lambda s: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        
    perform_validation = False
    log_results = False     
    
    optimal_class_threshold = 0.5
    best_val_metric = 0.0
    best_metric_iter = 0
    avg_train_logs = None
    n_iters_per_val_cycle = config["accumulate_batches"] * config["validate_every_n_updates"] 
    
    # wandb.watch(model, log='gradients', log_freq=config["validate_every_n_updates"])
    
    pbar = tqdm(total=config["max_updates"])
    if resume_config is None:
        n_update = -1
        n_iter = 0
    else:
        n_update = resume_config["n_update"]
        n_iter = resume_config["n_iter"]
        max_iters += n_iter   
        
        
    while n_iter < max_iters:
        for batch_ix, train_batch in enumerate(train_loader):
            n_iter += 1
            if n_iter>max_iters:
                break
            
            peptides, hla_pseudoseqs, imm_target = train_batch
            imm_target = imm_target.float().to(device)
            
            predictions = model(peptides, hla_pseudoseqs).view(-1)

            loss = loss_function(predictions, imm_target)
            train_loss_logs = {"train/BCE": loss.item()}
                
            if avg_train_logs is None:
                avg_train_logs = train_loss_logs.copy()
            else:
                for k in train_loss_logs:
                    avg_train_logs[k] += train_loss_logs[k]
            loss.backward()
            
            if n_iter % config['accumulate_batches'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                n_update += 1
                pbar.update(1)
                if n_update % config["validate_every_n_updates"] == 0:
                    perform_validation = True
                
                
            if perform_validation:
                perform_validation = False
                model.immunogenicity_model.eval()
                
                avg_val_logs = None
                val_predictions = []
                for ix, val_batch in enumerate(val_loader):
                    peptides, hla_pseudoseqs, imm_target = val_batch
                    imm_target = imm_target.float().to(device)
                    
                    predictions = model(peptides, hla_pseudoseqs).view(-1)

                    loss = loss_function(predictions, imm_target)
                    val_loss_logs = {"val/BCE": loss.item()}
                    
                    if avg_val_logs is None:
                        avg_val_logs = val_loss_logs.copy()
                    else:
                        for k in val_loss_logs:
                            avg_val_logs[k] += val_loss_logs[k]

                        
                    predictions = predictions.detach().cpu().tolist()
                    if not isinstance(predictions, list):
                        predictions = [predictions]
                    val_predictions.extend(predictions)             
                    
                targets = val_df["Target"].values
                
                val_classification_metrics = eval_classification_metrics(targets, val_predictions, 
                                                        is_logit=True, threshold=0.5)
                val_classification_metrics = {"val_class/"+k: v for k, v in val_classification_metrics.items()}
                
                for k in avg_val_logs:
                    avg_val_logs[k] /= len(val_loader)
                    
                    
                epoch_val_metric = val_classification_metrics["val_class/"+target_metric] 
                if epoch_val_metric>best_val_metric:
                    best_val_metric = epoch_val_metric
                    best_metric_iter = n_iter
                    torch.save(model.state_dict(), checkpoint_path)
                    resume_config = {"n_iter": n_iter, "n_update": n_update}
                    with open(os.path.join(checkpoints_folder, checkpoint_label+".json"), "w") as f:
                        json.dump(resume_config, f, indent=2)

                    
                log_results = True
                model.immunogenicity_model.train()
            
            
            if log_results:
                log_results = False
                for k in avg_train_logs:
                    avg_train_logs[k] /= (config['accumulate_batches'] * config["validate_every_n_updates"])
                
                current_lr = scheduler.get_last_lr()[0]
                logs = {"learning_rate": current_lr, 
                        "accumulated_batch": n_update}
                
                logs.update(avg_train_logs)
                logs.update(avg_val_logs)
                logs.update(val_classification_metrics)
                
                wandb.log(logs)      
                avg_train_logs = None
                
            if config.get("early_stopping", False):
                if (n_iter - best_metric_iter)/n_iters_per_val_cycle>config['patience']:
                    print("Val loss not improving, stopping training..\n\n")
                    break


    pbar.close()

    print(f"\nLoading state dict from {checkpoint_path}\n\n")
    model.load_state_dict(torch.load(checkpoint_path))
    model.immunogenicity_model.eval()
    
    
    print("Testing model..")
    test_predictions = []
    for ix, test_batch in tqdm(enumerate(test_loader)):
        peptides, hla_pseudoseqs, imm_target = test_batch
        imm_target = imm_target.float().to(device)
        
        predictions = model(peptides, hla_pseudoseqs).view(-1)
        predictions = predictions.detach().cpu().tolist()
        if not isinstance(predictions, list):
            predictions = [predictions]
        test_predictions.extend(predictions)

    test_predictions.extend([np.nan]*(len(test_df)-len(test_predictions)))

    
    test_df["Prediction"] = test_predictions
    test_df.to_csv(join(results_folder, f"test_predictions_{run_name}.csv"), index=False)
    test_df = test_df.dropna()
    targets = test_df["Target"].values
    test_metrics = eval_classification_metrics(targets, test_predictions, 
                                            is_logit=True, 
                                            threshold=optimal_class_threshold)
    test_metrics["class_threshold"] = optimal_class_threshold
    test_metrics = {"test_IEDB_class/"+k: v for k, v in test_metrics.items()}
    
    for k, v in test_metrics.items():
        wandb.run.summary[k] = v

    predicted_classes = (sigmoid(test_df["Prediction"])>optimal_class_threshold).astype(int).values
    wandb.log({"test_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=targets, preds=predicted_classes,
                        class_names=["Negative", "Positive"])})
    
    with open(join(results_folder, f"test_metrics_{run_name}.json"), "w") as f:
        json.dump(test_metrics, f)
                    
        
    dhlap_dset = SequencesInteractionDataset(dhlap_imm_df, hla_repr=config["hla_repr"])
    dhlap_loader = DataLoader(dhlap_dset, batch_size=config['batch_size'])
    
    test_predictions = []
    for ix, test_batch in tqdm(enumerate(dhlap_loader)):        
        peptides, hla_pseudoseqs = test_batch[:2]
        predictions = model(peptides, hla_pseudoseqs).view(-1)
        
        test_predictions.extend(predictions.detach().cpu().tolist())
        
    test_predictions.extend([np.nan]*(len(dhlap_imm_df)-len(test_predictions)))

    
    dhlap_imm_df["Prediction"] = test_predictions
    dhlap_imm_df.to_csv(join(results_folder, f"dhlap_test_predictions_{run_name}.csv"), index=False)

    targets = dhlap_imm_df["Label"].values
    
    test_metrics = eval_classification_metrics(targets, test_predictions, 
                                            is_logit=True, 
                                            threshold=optimal_class_threshold)
    test_metrics["class_threshold"] = optimal_class_threshold
    
    test_metrics = {"test_DHLAP_class/"+k: v for k, v in test_metrics.items()}
    
    for k, v in test_metrics.items():
        wandb.run.summary[k] = v
        
    predicted_classes = (sigmoid(dhlap_imm_df["Prediction"])>optimal_class_threshold).astype(int).values
    wandb.log({"test_DHLAP_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=targets, preds=predicted_classes,
                        class_names=["DHLAP_Negative", "DHLAP_Positive"])})
    
    with open(join(results_folder, f"test_metrics_dhlap_{run_name}.json"), "w") as f:
        json.dump(test_metrics, f)
        

    print("Training complete!")
    

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_number", type=int)
    parser.add_argument("--experiment_name", type=str, default="devel")
    parser.add_argument("--experiment_group", type=str, default="Class_plabel")
    parser.add_argument("--project_folder", type=str, default="/home/gvisona/Projects/SelfPeptides")
    parser.add_argument("--data_folder", type=str, default="/home/gvisona/Projects/SelfPeptides/outputs/Class_plabel/logical-sweep-6")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--force_restart", action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument("--min_subjects_tested", type=int, default=1)
    parser.add_argument("--hla_filter", default=None)
    
    parser.add_argument("--immunogenicity_df", type=str, 
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/Immunogenicity/Processed_TCell_IEDB_Beta_noPrior.csv")
    parser.add_argument("--dhlap_df", type=str, 
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/Immunogenicity/DHLAP_immunogenicity_data.csv")    
    parser.add_argument("--pseudo_seq_file", type=str, default="/home/gvisona/Projects/SelfPeptides/data/NetMHCpan_pseudoseq/MHC_pseudo.dat")
    parser.add_argument("--binding_affinity_df", type=str, 
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/Binding_Affinity/DHLAP_binding_affinity_data.csv")    
    
    parser.add_argument("--binding_model_config", type=str, default="/home/gvisona/Projects/SelfPeptides/trained_models/binding_model/config.json")
    parser.add_argument("--binding_model_checkpoint", type=str, default="/home/gvisona/Projects/SelfPeptides/trained_models/binding_model/checkpoints/001_checkpoint.pt")
    parser.add_argument("--sns_model_config", type=str, default="/home/gvisona/Projects/SelfPeptides/trained_models/sns_model/config.json")
    parser.add_argument("--sns_model_checkpoint", type=str, default="/home/gvisona/Projects/SelfPeptides/trained_models/sns_model/checkpoints/001_checkpoint.pt")
    
     
    parser.add_argument("--PMA_ln", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--PMA_num_heads", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.15)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--n_attention_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--mlp_input_dim", type=int, default=2560)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_num_layers", type=int, default=1)
    parser.add_argument("--transf_hidden_dim", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=1)
    
    
    # parser.add_argument("--peptide_emb_pooling", type=str, default="PMA")
    # parser.add_argument("--peptide_embedding_model", type=str, default="AATransformerEncoder")
    parser.add_argument("--hla_repr", type=str, default="Allele Pseudo-sequence")
    # parser.add_argument("--hla_embedding_model", type=str, default="shared")
    # parser.add_argument("--hla_emb_pooling", default=None)
    parser.add_argument("--pretrained_aa_embeddings", type=str,
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/aa_embeddings/normalized_learned_BA_AA_embeddings.npy")
    

    
    
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gen_negative_samples", action=argparse.BooleanOptionalAction, default=True)
    
    # parser.add_argument("--regression_weight", type=float, default=1.0)
    # parser.add_argument("--kl_weight", type=float, default=1.0)
    # parser.add_argument("--kl_loss_type", type=str, default="forward")
    parser.add_argument("--target_metric", type=str, default="F1")
    
    parser.add_argument("--max_updates", type=int, default=5)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--validate_every_n_updates", type=int, default=1)
    
    
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--trainval_df", type=str)
    parser.add_argument("--test_df", type=str)
    parser.add_argument("--augmentation_df", type=str)
    
    
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_batches", type=int, default=1)
    
    
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov_momentum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_frac", type=float, default=0.1)
    parser.add_argument("--ramp_up", type=float, default=0.1)
    parser.add_argument("--cool_down", type=float, default=0.8)


    parser.add_argument("--wandb_sweep", action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    config = vars(args)

    train(config=config)