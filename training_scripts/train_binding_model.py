import pandas as pd
import numpy as np
import argparse
from argparse import ArgumentParser 
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from os.path import exists, join
import json
from tqdm import tqdm

import sys
from selfpeptide.utils.training_utils import eval_classification_metrics, lr_schedule, warmup_constant_lr_schedule, get_class_weights, sigmoid
from selfpeptide.model.binding_affinity_classifier import Peptide_HLA_BindingClassifier
from selfpeptide.utils.data_utils import SequencesInteractionDataset, load_binding_affinity_dataframes


class WeightedBinding_Loss(nn.Module):
    def __init__(self, class_weights=[1.0, 1.0], device="cpu"):
        super().__init__()
        self.device = device
        self.class_weights = torch.tensor(class_weights).to(device)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")
        
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        weights = torch.gather(self.class_weights, 0, targets.long())
        loss = self.bce_logits(predictions.view(-1), targets)
        loss = torch.mean(loss * weights)
        return loss
        
        
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
    # torch.autograd.set_detect_anomaly(True) # DEUGGING
    if config["run_number"] is None:
        config["run_number"] = config["seed"]
    if run_name is None or len(run_name) < 1 or not config["wandb_sweep"]:
        run_name = str(config["run_number"])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not exists(config['project_folder']):
        raise ValueError("Project folder does not exist")
    
    output_folder = join(config['project_folder'], "outputs", config['experiment_group'], config['experiment_name'], run_name)
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    with open(join(output_folder, "config.json"), "w") as f:
        json.dump(dict(config), f)
    
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
        
    checkpoint_path = os.path.join(checkpoints_folder, checkpoint_fname)
    wandb.run.summary["checkpoints/Checkpoint_path"] = checkpoint_path
    checkpoint_label = checkpoint_fname.split(".")[0]

    train_ba_df, val_ba_df, test_ba_df = load_binding_affinity_dataframes(config)

    train_ba_df.to_csv(join(output_folder, f"train_df_{run_name}.csv"), index=False)
    val_ba_df.to_csv(join(output_folder, f"val_df_{run_name}.csv"), index=False)
    pos_weight, neg_weight = get_class_weights(train_ba_df, target_label="Label")
    
    wandb.run.summary["pos_weight"] = pos_weight
    wandb.run.summary["neg_weight"] = neg_weight
    
    train_dset = SequencesInteractionDataset(train_ba_df, hla_repr=config["hla_repr"])
    val_dset = SequencesInteractionDataset(val_ba_df, hla_repr=config["hla_repr"])
    test_dset = SequencesInteractionDataset(test_ba_df, hla_repr=config["hla_repr"])

    train_loader = DataLoader(train_dset, batch_size=config['batch_size'], drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=config['batch_size'], drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=config['batch_size'], drop_last=False)


    model = Peptide_HLA_BindingClassifier(config, device)
    model.to(device)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        
        
    if config["init_checkpoint"] is not None and resume_checkpoint_path is None:
        print("Initializing model using checkpoint {}".format(config["init_checkpoint"]))
        model.load_state_dict(torch.load(config["init_checkpoint"]))
        wandb.run.summary["checkpoints/Init_checkpoint"] = config["init_checkpoint"]
            
    if resume_checkpoint_path is not None:
        print("Resuming training from checkpoint {}".format(resume_checkpoint_path))
        model.load_state_dict(torch.load(resume_checkpoint_path))
        wandb.run.summary["checkpoints/Resuming_checkpoint"] = resume_checkpoint_path
        
    with open(join(output_folder, "architecture.txt"), "w") as f:
        f.write(model.__repr__())
        
    max_iters = config["max_updates"] * config["accumulate_batches"] + 1
    
    # binding_loss = CustomBindingLoss()WeightedBinding_Loss
    bce_loss = WeightedBinding_Loss(class_weights=[neg_weight, pos_weight], device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config.get("momentum", 0.9),
                                nesterov=config.get("nesterov_momentum", False),
                                weight_decay=config['weight_decay'])
    
    gen_train = iter(train_loader)    
    best_val_loss = np.inf
    best_loss_iter = 0
    
    avg_train_logs = None
    n_iters_per_val_cycle = config["accumulate_batches"] * config["validate_every_n_updates"]    
    
    
    # lr_lambda = lambda s: lr_schedule(s, min_frac=config['min_frac'], total_iters=config["max_updates"], 
    #                                   ramp_up=config['ramp_up'], cool_down=config['cool_down'])
    
    if resume_checkpoint_path is None and config["init_checkpoint"] is None:
        def lr_lambda(s): return warmup_constant_lr_schedule(s, min_frac=config['min_frac'], total_iters=config["max_updates"],
                                         ramp_up=config['ramp_up'])
    else:
        lr_lambda = lambda s: 1.0
        
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # model.eval() #TODO REMOVE
    n_update = -1
    perform_validation = False
    log_results = False
    for n_iter in tqdm(range(max_iters)):
        try:
            train_batch = next(gen_train)
        except StopIteration:
            gen_train = iter(train_loader)
            train_batch = next(gen_train)

        peptides, hla_pseudoseqs = train_batch[:2]
        batch_targets = train_batch[-1].float().to(device)
        
        predictions, _ = model(peptides, hla_pseudoseqs)
        loss = bce_loss(predictions, batch_targets)

        train_loss_logs = {"train/binding_BCE": loss.item()}
        
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
            if n_update % config["validate_every_n_updates"] == 0:
                perform_validation = True
        
            
            
        if perform_validation:
            avg_val_logs = None
            perform_validation = False
            model.eval()
            
            val_predictions = []
            for ix, val_batch in enumerate(val_loader):
                peptides, hla_pseudoseqs = val_batch[:2]
                batch_targets = val_batch[-1].float().to(device)
        
                predictions, _ = model(peptides, hla_pseudoseqs)
                loss = bce_loss(predictions, batch_targets)
                val_loss_logs = {"val/binding_BCE": loss.item()}

                
                if avg_val_logs is None:
                    avg_val_logs = val_loss_logs.copy()
                else:
                    for k in val_loss_logs:
                        avg_val_logs[k] += val_loss_logs[k]

                pred_ba = predictions.detach().cpu().squeeze().tolist()
                if not isinstance(pred_ba, list):
                    pred_ba = [pred_ba]
                val_predictions.extend(pred_ba)        
                
                
            for k in avg_val_logs:
                avg_val_logs[k] /= len(val_loader)
            
            targets = val_ba_df["Label"].values
            val_classification_metrics = eval_classification_metrics(targets[:len(val_predictions)], val_predictions, 
                                                    is_logit=True, threshold=0.5)
            val_classification_metrics = {"val_class/"+k: v for k, v in val_classification_metrics.items()}
            
            epoch_val_loss = avg_val_logs["val/binding_BCE"] 
            if epoch_val_loss<best_val_loss:
                best_val_loss = epoch_val_loss
                best_loss_iter = n_iter
                torch.save(model.state_dict(), checkpoint_path)
                resume_config = {"n_iter": n_iter, "n_update": n_update}
                with open(os.path.join(checkpoints_folder, checkpoint_label+".json"), "w") as f:
                    json.dump(resume_config, f, indent=2)
                    
                    
            log_results = True
            model.train()
            
            
        if log_results:
            log_results = False
            for k in avg_train_logs:
                avg_train_logs[k] /= (config['accumulate_batches'] * config["validate_every_n_updates"])
            
            current_lr = scheduler.get_last_lr()[0]
            val_train_difference = avg_val_logs["val/binding_BCE"] - avg_train_logs["train/binding_BCE"]
            logs = {"learning_rate": current_lr, 
                    "accumulated_batch": n_update, 
                    "train_val_difference": val_train_difference,
                    "best_val_loss": best_val_loss}
            
            logs.update(avg_train_logs)
            logs.update(avg_val_logs)
            logs.update(val_classification_metrics)
            
            wandb.log(logs)      
            avg_train_logs = None
            
            print("TEST ALL GOOD!")
            sys.exit(0)
            
        if config.get("early_stopping", False):
            if (n_iter - best_loss_iter)/n_iters_per_val_cycle>config['patience']:
                print("Val loss not improving, stopping training..\n\n")
                break
        
        
    print(os.path.exists(checkpoint_path))
    print(f"\nLoading state dict from {checkpoint_path}\n\n")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    print("Testing model..")
    test_predictions = []
    for ix, test_batch in tqdm(enumerate(test_loader)):
        peptides, hla_pseudoseqs = test_batch[:2]
        batch_targets = test_batch[-1].float().to(device)
        predictions, _ = model(peptides, hla_pseudoseqs)
        

        pred_ba = predictions.detach().cpu().squeeze().tolist()
        if not isinstance(pred_ba, list):
            pred_ba = [pred_ba]
        test_predictions.extend(pred_ba)
        
        
    test_predictions.extend([np.nan]*(len(test_ba_df)-len(test_predictions)))
    test_ba_df["Predicted Logits"] = test_predictions
    test_ba_df.to_csv(join(output_folder, f"test_predictions_{run_name}.csv"), index=False)
    test_ba_df = test_ba_df.dropna()
    
    targets = test_ba_df["Label"].values
    test_metrics = eval_classification_metrics(targets, test_ba_df["Predicted Logits"].values, 
                                               is_logit=True, threshold=0.5)
    
    predicted_classes = (sigmoid(test_ba_df["Predicted Logits"])>0.5).astype(int).values
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=targets, preds=predicted_classes,
                        class_names=[0, 1])})

    for k, v in test_metrics.items():
        wandb.run.summary[k] = v
    
    with open(join(output_folder, f"test_metrics_{run_name}.json"), "w") as f:
        json.dump(test_metrics, f)
        
        
    print("Training complete!")
    
    return model
    

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_number", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="devel_ba_separate")
    parser.add_argument("--experiment_group", type=str, default="BindingAffinity_DHLAP")
    parser.add_argument("--project_folder", type=str, default="/home/gvisona/Projects/SelfPeptides")
    parser.add_argument("--init_checkpoint", default=None)
    
    parser.add_argument("--PMA_num_heads", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.15)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--n_attention_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=2)
    
    parser.add_argument("--regression_weight", type=float, default=1.0)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--hla_repr", type=str, default="Allele Pseudo-sequence")
    parser.add_argument("--transf_hidden_dim", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_num_layers", type=int, default=1)
    
    
    parser.add_argument("--binding_affinity_df", type=str, 
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/Binding_Affinity/DHLAP_binding_affinity_data.csv")    
    parser.add_argument("--pseudo_seq_file", type=str, default="/home/gvisona/Projects/SelfPeptides/data/NetMHCpan_pseudoseq/MHC_pseudo.dat")
    # parser.add_argument("--pretrained_aa_embeddings", type=str,
    #                     default="/home/gvisona/Projects/SelfPeptides/processed_data/aa_embeddings/normalized_learned_BA_AA_embeddings.npy")

    parser.add_argument("--max_updates", type=int, default=5)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--validate_every_n_updates", type=int, default=1)
    
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.1)
    
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_batches", type=int, default=1)
    
    
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--nesterov_momentum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_frac", type=float, default=0.01)
    parser.add_argument("--ramp_up", type=float, default=0.3)
    parser.add_argument("--cool_down", type=float, default=0.6)
    
    
    parser.add_argument("--wandb_sweep", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    
    config = vars(args)

    train(config=config)