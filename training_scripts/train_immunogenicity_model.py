
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
import sys
from os.path import exists, join
from scipy.stats import beta
from tqdm import tqdm
from selfpeptide.utils.data_utils import load_immunogenicity_dataframes_jointseqs, BetaDistributionDataset, SequencesInteractionDataset
from selfpeptide.utils.training_utils import find_optimal_class_threshold, BetaChernoffDistance, DIR_Weighted_RegressionLoss, BetaChernoffDistance_OptimizePrecision
from selfpeptide.utils.training_utils import lr_schedule, warmup_constant_lr_schedule, get_LDS_loss_weights
from selfpeptide.utils.training_utils import eval_beta_metrics, eval_regression_metrics, eval_classification_metrics
from selfpeptide.utils.beta_distr_utils import *
from selfpeptide.utils.model_utils import load_binding_model, load_sns_model
from selfpeptide.model.immunogenicity_predictor import JointPeptidesNetwork_Beta


from selfpeptide.utils.training_utils import BetaKLDivergence_OptimizePrecision



class BetaDistance_with_Constraints(nn.Module):
    def __init__(self, config={}, device="cpu", **kwargs):
        super().__init__()
        
        self.chernoff_distance = BetaChernoffDistance_OptimizePrecision(config=config, device=device, **kwargs)
        self.beta_kl = BetaKLDivergence_OptimizePrecision(config=config, device=device, **kwargs)
        self.kl_weight = 0.1
        self.use_posterior = config.get("use_posterior_mean", False)
        self.constraints_weight = config.get("constraints_weight", 1.0)
        
        self.regression_loss = DIR_Weighted_RegressionLoss(config=config, device=device, **kwargs)


    def forward(self, predictions, target_alphas, target_betas):
        beta_chernoff_loss, chernoff_logs = self.chernoff_distance(predictions, target_alphas, target_betas)
        beta_kl_loss, logs = self.beta_kl(predictions, target_alphas, target_betas)
        
        binding_scores = predictions[:, 0]

        if self.use_posterior:
            means = predictions[:,2]
        else:
            means = predictions[:,1]
        precisions = predictions[:, 3]
        
        alphas, betas, variances, modes = beta_distr_params_from_mean_precision(means, precisions)
        target_means, target_precisions, target_variances, target_modes = beta_distr_params_from_alpha_beta(target_alphas, target_betas)
        constraints_loss = torch.sum(torch.clamp(means.view(-1)-binding_scores.view(-1), min=0.0))
        
        regr_loss, regr_loss_logs = self.regression_loss(predictions, target_alphas, target_betas)
        logs.update(regr_loss_logs)
        logs.update(chernoff_logs)
        # loss = beta_chernoff_loss + self.constraints_weight * constraints_loss
        loss =  self.kl_weight * beta_kl_loss + regr_loss + self.constraints_weight * constraints_loss
        logs["loss"] = loss.item()
        
        logs["constraints_loss"] = constraints_loss.item()
        logs["weighted_constraints_loss"] = self.constraints_weight * constraints_loss.item()
        
        logs["beta_mean_mAE"] = torch.mean(torch.abs(means-target_means))
        logs["beta_precisions_mAE"] = torch.mean(torch.abs(precisions-target_precisions))
        
        return loss, logs
    
    
    
    
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
    train_df, val_df, test_df, dhlap_imm_df = load_immunogenicity_dataframes_jointseqs(config)
    
    
    with open(join(output_folder, "config.json"), "w") as f:
        json.dump(dict(config), f)
        
    checkpoint_path = os.path.join(output_folder, "checkpoint.pt")
    
    optimal_mean_class_threshold = find_optimal_class_threshold(train_df, "Distr. Mean", target_metric=config.get("target_class_metric", "MCC"))
    optimal_mode_class_threshold = find_optimal_class_threshold(train_df, "Distr. Mode", target_metric=config.get("target_class_metric", "MCC"))
    # target_metric = config.get("target_metric", "AUPRC")
    wandb.run.summary["mean_class_threshold"] = optimal_mean_class_threshold
    wandb.run.summary["mode_class_threshold"] = optimal_mode_class_threshold
    
    
    try_to_overfit = config.get("try_to_overfit", False)
    
    
    train_dset = BetaDistributionDataset(train_df, hla_repr=["Allele Pseudo-sequence", "Allele Protein sequence"])
    val_dset = BetaDistributionDataset(val_df, hla_repr=["Allele Pseudo-sequence", "Allele Protein sequence"])
    test_dset = BetaDistributionDataset(test_df, hla_repr=["Allele Pseudo-sequence", "Allele Protein sequence"])
    
    if try_to_overfit:
        # n_samples = config['batch_size'] * 64
        # train_dset = BetaDistributionDataset(train_df.iloc[:n_samples], hla_repr=["Allele Pseudo-sequence", "Allele Protein sequence"])
        train_loader = DataLoader(train_dset, batch_size=config['batch_size'], drop_last=True, shuffle=False)

    else:
        train_loader = DataLoader(train_dset, batch_size=config['batch_size'], drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=config['batch_size'], drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=config['batch_size'], drop_last=False)
    
    print("Loading model")
    with open(config["binding_model_config"], "r") as f:
        binding_model_config = json.load(f)
    if os.path.exists("/home/gvisona/Projects"):
        for k in binding_model_config.keys():
            if not isinstance(binding_model_config[k], str):
                continue
            if "/home/gvisona/SelfPeptides" in binding_model_config[k]:
                binding_model_config[k] = binding_model_config[k].replace("/home/gvisona/SelfPeptides", "/home/gvisona/Projects/SelfPeptides")
            if "/fast/gvisona/SelfPeptides" in binding_model_config[k]:
                binding_model_config[k] = binding_model_config[k].replace("/fast/gvisona/SelfPeptides", "/home/gvisona/Projects/SelfPeptides")
    
    model = JointPeptidesNetwork_Beta(config, binding_model_config, 
                             binding_checkpoint=config["binding_model_checkpoint"], 
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
    
    bins, bin_weights, emp_label_dist, eff_label_dist = get_LDS_loss_weights(train_df["Distr. Mean"], 
                                                                                config["IR_n_bins"], 
                                                                                config["LDS_kernel"], 
                                                                                config["LDS_ks"], 
                                                                                config["LDS_sigma"])

    bins = torch.tensor(bins).to(device)
    bin_weights = torch.tensor(bin_weights).to(device)
        
    # beta_loss = BetaChernoffDistance(config, device=device, bins=bins, bin_weights=bin_weights) 
    beta_loss = BetaDistance_with_Constraints(config, device=device, bins=bins, bin_weights=bin_weights) 
    
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
    
    
    # optimizer = torch.optim.SGD(model.immunogenicity_model.parameters(), lr=config['lr'], momentum=config.get("momentum", 0.9),
    #                             nesterov=config.get("nesterov_momentum", False),
    #                             weight_decay=config['weight_decay'])
    optimizer = torch.optim.AdamW(model.immunogenicity_model.parameters(), lr=config['lr'], 
                            weight_decay=config['weight_decay'])

    if resume_checkpoint_path is None and config.get("init_checkpoint", None) is None:
        def lr_lambda(s): return warmup_constant_lr_schedule(s, min_frac=config['min_frac'], total_iters=config["max_updates"],
                                         ramp_up=config['ramp_up'])
    else:
        lr_lambda = lambda s: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        
    perform_validation = False
    log_results = False     
    
    use_posterior_mean = config.get("use_posterior_mean", False)
    best_val_metric = np.inf
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
        
    early_stopping = False
    while n_iter < max_iters:
        for batch_ix, train_batch in enumerate(train_loader):
            n_iter += 1
            if n_iter>max_iters:
                break
            
            peptides, hla_pseudoseqs, hla_prots = train_batch[:3]
            imm_alpha, imm_beta, imm_target = train_batch[3:]
            
            imm_alpha = imm_alpha.float().to(device)
            imm_beta = imm_beta.float().to(device)
            imm_target = imm_target.float().to(device)
            
            predictions = model(peptides, hla_pseudoseqs, hla_prots)

            loss, train_loss_logs = beta_loss(predictions, imm_alpha, imm_beta)
            train_loss_logs = {"train/"+str(k): v for k, v in train_loss_logs.items()}
                
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
                
                
            if perform_validation and not try_to_overfit:
                perform_validation = False
                model.immunogenicity_model.eval()
                
                avg_val_logs = None
                val_pred_means = []
                val_pred_precisions = []
                for ix, val_batch in enumerate(val_loader):
                    peptides, hla_pseudoseqs, hla_prots = val_batch[:3]
                    imm_alpha, imm_beta, imm_target = val_batch[3:]

                    imm_alpha = imm_alpha.float().to(device)
                    imm_beta = imm_beta.float().to(device)
                    imm_target = imm_target.float().to(device)
                    
                    predictions = model(peptides, hla_pseudoseqs, hla_prots)

                    loss, val_loss_logs = beta_loss(predictions, imm_alpha, imm_beta)
                    val_loss_logs = {"val/"+str(k): v for k, v in val_loss_logs.items()}
                    
                    if avg_val_logs is None:
                        avg_val_logs = val_loss_logs.copy()
                    else:
                        for k in val_loss_logs:
                            avg_val_logs[k] += val_loss_logs[k]
                    
            
                    if use_posterior_mean:
                        pred_means = predictions[:,2]         
                    else:
                        pred_means = predictions[:,1]                        
                    pred_precisions = predictions[:,3]
                    
                        
                    pred_means = pred_means.detach().cpu().tolist()
                    if not isinstance(pred_means, list):
                        pred_means = [pred_means]
                    val_pred_means.extend(pred_means)        
                    
                    pred_precisions = pred_precisions.detach().cpu().tolist()
                    if not isinstance(pred_precisions, list):
                        pred_precisions = [pred_precisions]
                    val_pred_precisions.extend(pred_precisions)        
                
                val_pred_means = np.array(val_pred_means)
                val_pred_precisions = np.array(val_pred_precisions)
                val_pred_alphas, val_pred_betas, val_pred_vars, val_pred_modes = beta_distr_params_from_mean_precision(val_pred_means, val_pred_precisions)

                class_targets = val_df["Target"].values
                val_mean_classification_metrics = eval_classification_metrics(class_targets, val_pred_means, 
                                                        is_logit=False, threshold=optimal_mean_class_threshold)
                val_mean_classification_metrics = {"val_mean_class/"+k: v for k, v in val_mean_classification_metrics.items()}
                
                val_mode_classification_metrics = eval_classification_metrics(class_targets, val_pred_modes, 
                                                        is_logit=False, threshold=optimal_mode_class_threshold)
                val_mode_classification_metrics = {"val_mode_class/"+k: v for k, v in val_mode_classification_metrics.items()}
                
                regression_mean_targets = val_df["Distr. Mean"].values
                val_regression_mean_metrics = eval_regression_metrics(regression_mean_targets, val_pred_means)
                val_regression_mean_metrics = {"val_mean_regr/"+k: v for k, v in val_regression_mean_metrics.items()}
                
                ixs_low_targets = np.where(regression_mean_targets<0.5)[0]
                ixs_high_targets = np.where(regression_mean_targets>=0.5)[0]
                val_regression_mean_metrics_low = eval_regression_metrics(regression_mean_targets[ixs_low_targets], val_pred_means[ixs_low_targets])
                val_regression_mean_metrics_low = {"val_mean_regr_low/"+k: v for k, v in val_regression_mean_metrics_low.items()}
                val_regression_mean_metrics_high = eval_regression_metrics(regression_mean_targets[ixs_high_targets], val_pred_means[ixs_high_targets])
                val_regression_mean_metrics_high = {"val_mean_regr_high/"+k: v for k, v in val_regression_mean_metrics_high.items()}
                
                regression_mode_targets = val_df["Distr. Mode"].values
                val_regression_mode_metrics = eval_regression_metrics(regression_mode_targets, val_pred_modes)
                val_regression_mode_metrics = {"val_mode_regr/"+k: v for k, v in val_regression_mode_metrics.items()}
                
                                
                ixs_low_targets = np.where(regression_mode_targets<0.5)[0]
                ixs_high_targets = np.where(regression_mode_targets>=0.5)[0]
                val_regression_mode_metrics_low = eval_regression_metrics(regression_mode_targets[ixs_low_targets], val_pred_modes[ixs_low_targets])
                val_regression_mode_metrics_low = {"val_mode_regr_low/"+k: v for k, v in val_regression_mode_metrics_low.items()}
                val_regression_mode_metrics_high = eval_regression_metrics(regression_mode_targets[ixs_high_targets], val_pred_modes[ixs_high_targets])
                val_regression_mode_metrics_high = {"val_mode_regr_high/"+k: v for k, v in val_regression_mode_metrics_high.items()}
                
                target_means = val_df["Distr. Mean"].values
                target_precisions = val_df["Distr. Precision"].values
                val_beta_metrics = eval_beta_metrics(target_means, target_precisions, val_pred_means, val_pred_precisions)
                val_beta_metrics = {"val_beta/"+k: v for k, v in val_beta_metrics.items()}
                
                for k in avg_val_logs:
                    avg_val_logs[k] /= len(val_loader)
                    
                    
                epoch_val_metric = avg_val_logs["val/loss"] 
                if epoch_val_metric<best_val_metric:
                    best_val_metric = epoch_val_metric
                    best_metric_iter = n_iter
                    torch.save(model.state_dict(), checkpoint_path)
                    resume_config = {"n_iter": n_iter, "n_update": n_update}
                    with open(os.path.join(checkpoints_folder, checkpoint_label+".json"), "w") as f:
                        json.dump(resume_config, f, indent=2)

                    
                log_results = True
                model.immunogenicity_model.train()
            
            
            if try_to_overfit and n_iter % n_iters_per_val_cycle==0:
                log_results = True
            
            if log_results:
                log_results = False
                for k in avg_train_logs:
                    avg_train_logs[k] /= (config['accumulate_batches'] * config["validate_every_n_updates"])
                
                current_lr = scheduler.get_last_lr()[0]
                logs = {"learning_rate": current_lr, 
                        "accumulated_batch": n_update}
                
                logs.update(avg_train_logs)
                if not try_to_overfit:
                    logs.update(avg_val_logs)
                    # logs.update(val_regression_metrics)
                    logs.update(val_mean_classification_metrics)
                    logs.update(val_mode_classification_metrics)
                    logs.update(val_beta_metrics)
                    logs.update(val_regression_mean_metrics)
                    logs.update(val_regression_mode_metrics)
                    
                    logs.update(val_regression_mean_metrics_low)
                    logs.update(val_regression_mean_metrics_high)
                    logs.update(val_regression_mode_metrics_low)
                    logs.update(val_regression_mode_metrics_high)
                
                
                wandb.log(logs)      
                avg_train_logs = None
                
            if config.get("early_stopping", False) and not try_to_overfit:
                if (n_iter - best_metric_iter)/n_iters_per_val_cycle>config['patience']:
                    print("Val loss not improving, stopping training..\n\n")
                    early_stopping = True
                    break
                
            
        if early_stopping:
            break
                


    pbar.close()
    
    if try_to_overfit:
        print("Overfitting training complete!")
        sys.exit(0)

    print(f"\nLoading state dict from {checkpoint_path}\n\n")
    model.load_state_dict(torch.load(checkpoint_path))
    model.immunogenicity_model.eval()
    
    
    print("Testing model..")
    test_pred_means = []
    test_pred_precisions = []
    for ix, test_batch in tqdm(enumerate(test_loader)):
        peptides, hla_pseudoseqs, hla_prots = test_batch[:3]
        imm_alpha, imm_beta, imm_target = test_batch[3:]

        imm_alpha = imm_alpha.float().to(device)
        imm_beta = imm_beta.float().to(device)
        imm_target = imm_target.float().to(device)
        
        predictions = model(peptides, hla_pseudoseqs, hla_prots)
        if use_posterior_mean:
            pred_means = predictions[:,2]         
        else:
            pred_means = predictions[:,1]          
                      
        pred_precisions = predictions[:,3]
        
        pred_means = pred_means.detach().cpu().tolist()
        if not isinstance(pred_means, list):
            pred_means = [pred_means]
        test_pred_means.extend(pred_means)
        
        pred_precisions = pred_precisions.detach().cpu().tolist()
        if not isinstance(pred_precisions, list):
            pred_precisions = [pred_precisions]
        test_pred_precisions.extend(pred_precisions)
        
        
    test_pred_means.extend([np.nan]*(len(test_df)-len(test_pred_means)))
    test_pred_precisions.extend([np.nan]*(len(test_df)-len(test_pred_precisions)))
    test_df = test_df.dropna()
    
    test_pred_alphas, test_pred_betas, test_pred_vars, test_pred_modes = beta_distr_params_from_mean_precision(test_pred_means, test_pred_precisions)

    test_df["Prediction Distr. Mean"] = test_pred_means
    test_df["Prediction Distr. Precision"] = test_pred_precisions
    
    test_df.to_csv(join(results_folder, f"test_predictions_{run_name}.csv"), index=False)

    class_targets = test_df["Target"].values
    test_mean_classification_metrics = eval_classification_metrics(class_targets, test_pred_means, 
                                            is_logit=False, threshold=optimal_mean_class_threshold)
    test_mean_classification_metrics = {"test_IEDB_mean_class/"+k: v for k, v in test_mean_classification_metrics.items()}
    
    test_mode_classification_metrics = eval_classification_metrics(class_targets, test_pred_modes, 
                                            is_logit=False, threshold=optimal_mode_class_threshold)
    test_mode_classification_metrics = {"test_IEDB_mode_class/"+k: v for k, v in test_mode_classification_metrics.items()}
    
    regression_mean_targets = test_df["Distr. Mean"].values
    test_regression_mean_metrics = eval_regression_metrics(regression_mean_targets, test_pred_means)
    test_regression_mean_metrics = {"test_IEDB_mean_regr/"+k: v for k, v in test_regression_mean_metrics.items()}
    
    regression_mode_targets = test_df["Distr. Mode"].values
    test_regression_mode_metrics = eval_regression_metrics(regression_mode_targets, test_pred_modes)
    test_regression_mode_metrics = {"test_IEDB_mode_regr/"+k: v for k, v in test_regression_mode_metrics.items()}
    
    target_means = test_df["Distr. Mean"].values
    target_precisions = test_df["Distr. Precision"].values
    test_beta_metrics = eval_beta_metrics(target_means, target_precisions, test_pred_means, test_pred_precisions)
    test_beta_metrics = {"test_IEDB_beta/"+k: v for k, v in test_beta_metrics.items()}
                
    
    for k, v in test_mean_classification_metrics.items():
        wandb.run.summary[k] = v
    for k, v in test_mode_classification_metrics.items():
        wandb.run.summary[k] = v
    for k, v in test_regression_mean_metrics.items():
        wandb.run.summary[k] = v
    for k, v in test_regression_mode_metrics.items():
        wandb.run.summary[k] = v
    
    predicted_mean_classes = (test_df["Prediction Distr. Mean"]>optimal_mean_class_threshold).astype(int).values
    wandb.log({"test_mean_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=class_targets, preds=predicted_mean_classes,
                        class_names=["IEDB Mean Negative", "IEDB Mean Positive"])})
    
    predicted_mode_classes = (test_pred_modes>optimal_mode_class_threshold).astype(int)
    wandb.log({"test_mode_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=class_targets, preds=predicted_mode_classes,
                        class_names=["IEDB Mode Negative", "IEDB Mode Positive"])})
    
    # with open(join(results_folder, f"test_metrics_{run_name}.json"), "w") as f:
    #     json.dump(test_metrics, f)
                    
        
        
        
        
        
        
    # dhlap_dset = SequencesInteractionDataset(dhlap_imm_df, hla_repr=["Allele Pseudo-sequence", "Allele Protein sequence"])
    # dhlap_loader = DataLoader(dhlap_dset, batch_size=config['batch_size'])
    
    # test_pred_means = []
    # test_pred_precisions = []
    # for ix, test_batch in tqdm(enumerate(dhlap_loader)):        
    #     peptides, hla_pseudoseqs, hla_prots = test_batch[:3]
    #     predictions = model(peptides, hla_pseudoseqs, hla_prots)
        
    #     if use_posterior_mean:
    #         pred_means = predictions[:,2]         
    #     else:
    #         pred_means = predictions[:,1]          
                      
    #     pred_precisions = predictions[:,3]
        
    #     pred_means = pred_means.detach().cpu().tolist()
    #     if not isinstance(pred_means, list):
    #         pred_means = [pred_means]
    #     test_pred_means.extend(pred_means)
        
    #     pred_precisions = pred_precisions.detach().cpu().tolist()
    #     if not isinstance(pred_precisions, list):
    #         pred_precisions = [pred_precisions]
    #     test_pred_precisions.extend(pred_precisions)
        

    # test_pred_means.extend([np.nan]*(len(dhlap_imm_df)-len(test_pred_means)))
    # test_pred_precisions.extend([np.nan]*(len(dhlap_imm_df)-len(test_pred_precisions)))
    
    # test_pred_alphas, test_pred_betas, test_pred_vars, test_pred_modes = beta_distr_params_from_mean_precision(test_pred_means, test_pred_precisions)
    
    # dhlap_imm_df["Prediction Distr. Mean"] = test_pred_means
    # dhlap_imm_df["Prediction Distr. Precision"] = test_pred_precisions
    
    
    # predicted_mean_classes = (dhlap_imm_df["Prediction Distr. Mean"]>optimal_mean_class_threshold).astype(int).values
    # predicted_mode_classes = (test_pred_modes>optimal_mode_class_threshold).astype(int)

    # dhlap_imm_df["Predicted Class (Mean)"] = predicted_mean_classes
    # dhlap_imm_df["Predicted Class (Mode)"] = predicted_mode_classes

    # dhlap_imm_df.to_csv(join(results_folder, f"dhlap_test_predictions_{run_name}.csv"), index=False)

    # class_targets = dhlap_imm_df["Label"].values
    
    # test_mean_classification_metrics = eval_classification_metrics(class_targets, test_pred_means, 
    #                                         is_logit=False, threshold=optimal_mean_class_threshold)
    # test_mean_classification_metrics = {"test_DHLAP_mean_class/"+k: v for k, v in test_mean_classification_metrics.items()}
    
    # test_mode_classification_metrics = eval_classification_metrics(class_targets, test_pred_modes, 
    #                                         is_logit=False, threshold=optimal_mode_class_threshold)
    # test_mode_classification_metrics = {"test_DHLAP_mode_class/"+k: v for k, v in test_mode_classification_metrics.items()}
    

    # for k, v in test_mean_classification_metrics.items():
    #     wandb.run.summary[k] = v
    # for k, v in test_mode_classification_metrics.items():
    #     wandb.run.summary[k] = v

    
    # wandb.log({"test_DHLAP_mean_conf_mat" : wandb.plot.confusion_matrix(probs=None,
    #                     y_true=class_targets, preds=predicted_mean_classes,
    #                     class_names=["DHLAP Mean Negative", "DHLAP Mean Positive"])})
    
    # wandb.log({"test_DHLAP_mode_conf_mat" : wandb.plot.confusion_matrix(probs=None,
    #                 y_true=class_targets, preds=predicted_mode_classes,
    #                 class_names=["DHLAP Mode Negative", "DHLAP Mode Positive"])})

    
    ## with open(join(results_folder, f"test_metrics_dhlap_{run_name}.json"), "w") as f:
    ##     json.dump(test_metrics, f)
        
    print("Training complete!")
    
    
    
    
    

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_number", type=int)
    parser.add_argument("--experiment_name", type=str, default="BetaRegr_Overfitting")
    parser.add_argument("--experiment_group", type=str, default="Beta_regs_js")
    parser.add_argument("--project_folder", type=str, default="/home/gvisona/Projects/SelfPeptides")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--force_restart", action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument("--min_subjects_tested", type=int, default=1)
    parser.add_argument("--hla_filter", default=None)
    
    parser.add_argument("--immunogenicity_df", type=str, 
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/Immunogenicity/Processed_TCell_IEDB_beta_summed.csv")
    parser.add_argument("--dhlap_df", type=str, 
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/Immunogenicity/DHLAP_immunogenicity_data.csv")    
    parser.add_argument("--pseudo_seq_file", type=str, default="/home/gvisona/Projects/SelfPeptides/processed_data/HLA_embeddings/HLA_pseudoseqs_T5/hla_pseudoseq_mapping.csv")
    parser.add_argument("--hla_prot_seq_file", type=str, default="/home/gvisona/Projects/SelfPeptides/processed_data/HLA_embeddings/HLA_proteins_T5/hla_proteins_mapping.csv")
    
    parser.add_argument("--binding_model_config", type=str, default="/home/gvisona/Projects/SelfPeptides/trained_models/BindingModels/floral-sweep-3/config.json")
    parser.add_argument("--binding_model_checkpoint", type=str, 
                        default="/home/gvisona/Projects/SelfPeptides/trained_models/BindingModels/floral-sweep-3/checkpoints/001_checkpoint.pt")

     
    parser.add_argument("--dropout_p", type=float, default=0.0)
    # parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--mlp_input_dim", type=int, default=1025)
    parser.add_argument("--mlp_hidden_dim", type=int, default=2048)
    parser.add_argument("--mlp_num_layers", type=int, default=4)
    parser.add_argument("--mlp_output_dim", type=int, default=512)
    parser.add_argument("--imm_regression_hidden_dim", type=int, default=2048)
    parser.add_argument("--kl_loss_type", type=str, default="forward")



    parser.add_argument("--max_updates", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--validate_every_n_updates", type=int, default=5)
    
    
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--trainval_df", type=str)
    parser.add_argument("--test_df", type=str)
    
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulate_batches", type=int, default=4)
    
    parser.add_argument("--IR_n_bins", type=int, default=50)
    
    parser.add_argument("--LDS_kernel", type=str, default="triang")
    parser.add_argument("--LDS_ks", type=int, default=11)
    parser.add_argument("--LDS_sigma", type=int, default=1.0)
    parser.add_argument("--loss_weights", type=str, default="LDS_weights")
    parser.add_argument("--constraints_weight", type=int, default=0.00)
    
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    # parser.add_argument("--momentum", type=float, default=0.9)
    # parser.add_argument("--nesterov_momentum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_frac", type=float, default=0.1)
    parser.add_argument("--ramp_up", type=float, default=0.1)
    parser.add_argument("--cool_down", type=float, default=0.8)


    parser.add_argument("--use_posterior_mean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--try_to_overfit", action=argparse.BooleanOptionalAction, default=False)
    
    parser.add_argument("--wandb_sweep", action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    config = vars(args)

    train(config=config)