# Load model directly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os.path import join, exists
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import argparse
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import wandb
from selfpeptide.utils.training_utils import warmup_constant_lr_schedule
from tqdm import tqdm
from copy import deepcopy



class PeptidesDataset(Dataset):
    def __init__(self, peptides):
        super().__init__()
        self.peptides = peptides
        
    def __len__(self):
        return len(self.peptides)
    
    def __getitem__(self, ix):
        return self.peptides[ix]


def mask_tokenized_inputs(tokenized_dict, mlm_fraction=0.15, mask_token_id=32):
    out_dict = deepcopy(tokenized_dict.copy())
    lengths = tokenized_dict["attention_mask"].sum(axis=1)
    
    row_ixs = []
    col_ixs = []
    
    for i, l in enumerate(lengths):
        n_masked_tokens = int(mlm_fraction*l)
        row_ixs.extend([i]*n_masked_tokens)
        col_ixs.append(np.random.choice(list(range(l)), size=n_masked_tokens, replace=False))
    col_ixs = np.concatenate(col_ixs)
    row_ixs = np.array(row_ixs)
    out_dict["input_ids"][row_ixs, col_ixs] = mask_token_id
    return out_dict

    
    
def finetune_model(config=None, init_wandb=True):
    if init_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=config['experiment_name'],
            
            # track hyperparameters and run metadata
            config=config
        )
    
        config = wandb.config
    
    run_name = wandb.run.name
    # run_name = None
    # torch.autograd.set_detect_anomaly(True) # DEUGGING
    if config["run_number"] is None:
        config["run_number"] = config["seed"]
    if run_name is None or len(run_name) < 1 or not config["wandb_sweep"]:
        run_name = str(config["run_number"])
        
    if not exists(config['project_folder']):
        raise ValueError("Project folder does not exist")
    
    output_folder = join(config['project_folder'], "outputs", config['experiment_group'], config['experiment_name'], run_name)
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    with open(join(output_folder, "config.json"), "w") as f:
        json.dump(dict(config), f)
    
    checkpoints_folder = join(output_folder, "checkpoints")
    os.makedirs(checkpoints_folder, exist_ok=True)    
        
    checkpoint_fname = "001_checkpoint.pt"    
    checkpoint_path = os.path.join(checkpoints_folder, checkpoint_fname)
    # wandb.run.summary["checkpoints/Checkpoint_path"] = checkpoint_path
    checkpoint_label = checkpoint_fname.split(".")[0]

    peptides_set = set()
    for dname in config["peptides_dataframes"]:
        dpath = join(config["data_folder"], dname)
        df = pd.read_csv(dpath)
        peptides_set.update(df["Peptide"].values)
    n_peptides = len(peptides_set)
    print(f"Total number of peptides: {n_peptides}")
    peptides_set = sorted(list(peptides_set))


    if config.get("test_run", False):
        peptides_set = peptides_set[:10000] # TODO REMOVE FOR DEBUGGING
    

    trainval_set, test_set = train_test_split(peptides_set, test_size=config["test_size"], random_state=config["seed"])
    train_set, val_set = train_test_split(trainval_set, test_size=config["val_size"], random_state=config["seed"])

    train_dset = PeptidesDataset(train_set)
    val_dset = PeptidesDataset(val_set)
    test_dset = PeptidesDataset(test_set)

    train_loader = DataLoader(train_dset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=config["batch_size"], drop_last=False)
    test_loader = DataLoader(test_dset, batch_size=config["batch_size"], drop_last=False)
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])
    model = AutoModelForMaskedLM.from_pretrained(config["pretrained_model"])
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config.get("momentum", 0.9),
                                nesterov=config.get("nesterov_momentum", False),
                                weight_decay=config['weight_decay'])
    
    max_iters = config["max_updates"] * config["accumulate_batches"] + 1
    gen_train = iter(train_loader)    
    best_val_loss = np.inf
    best_loss_iter = 0
    
    avg_train_logs = None
    n_iters_per_val_cycle = config["accumulate_batches"] * config["validate_every_n_updates"]    
    
    
    
    def lr_lambda(s): return warmup_constant_lr_schedule(s, min_frac=config['min_frac'], total_iters=config["max_updates"],
                                         ramp_up=config['ramp_up'])
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    n_update = -1
    perform_validation = False
    log_results = False
    for n_iter in tqdm(range(max_iters)):
        try:
            train_batch = next(gen_train)
        except StopIteration:
            gen_train = iter(train_loader)
            train_batch = next(gen_train)

        encoded_batch = tokenizer(train_batch, return_tensors="pt", padding=True)#.to(device)
        masked_batch = mask_tokenized_inputs(encoded_batch, mlm_fraction=config["mlm_fraction"], mask_token_id=tokenizer.mask_token_id).to(device)
        labels = torch.where(masked_batch.input_ids == tokenizer.mask_token_id, encoded_batch["input_ids"], -100)
        outputs = model(**masked_batch, labels=labels, output_hidden_states=False)
        loss = outputs.loss
        train_loss_logs = {"train/loss": loss.item()}

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
            
            for ix, val_batch in enumerate(val_loader):
                encoded_batch = tokenizer(val_batch, return_tensors="pt", padding=True)#.to(device)
                masked_batch = mask_tokenized_inputs(encoded_batch, mlm_fraction=config["mlm_fraction"], mask_token_id=tokenizer.mask_token_id).to(device)
                labels = torch.where(masked_batch.input_ids == tokenizer.mask_token_id, encoded_batch["input_ids"], -100)
                outputs = model(**masked_batch, labels=labels, output_hidden_states=False)
                loss = outputs.loss
                val_loss_logs = {"val/loss": loss.item()}

                
                if avg_val_logs is None:
                    avg_val_logs = val_loss_logs.copy()
                else:
                    for k in val_loss_logs:
                        avg_val_logs[k] += val_loss_logs[k]
                
            for k in avg_val_logs:
                avg_val_logs[k] /= len(val_loader)
            
            epoch_val_loss = avg_val_logs["val/loss"] 
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
            val_train_difference = avg_val_logs["val/loss"] - avg_train_logs["train/loss"]
            logs = {"learning_rate": current_lr, 
                    "accumulated_batch": n_update, 
                    "train_val_difference": val_train_difference,
                    "best_val_loss": best_val_loss}
            
            logs.update(avg_train_logs)
            logs.update(avg_val_logs)

            wandb.log(logs)      
            avg_train_logs = None
            
        # if config.get("early_stopping", False):
        #     if (n_iter - best_loss_iter)/n_iters_per_val_cycle>config['patience']:
        #         print("Val loss not improving, stopping training..\n\n")
        #         break
        
    print(os.path.exists(checkpoint_path))
    print(f"\nLoading state dict from {checkpoint_path}\n\n")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    avg_test_logs = None
    print("Testing model..")
    for ix, test_batch in tqdm(enumerate(test_loader)):
        encoded_batch = tokenizer(test_batch, return_tensors="pt", padding=True)#.to(device)
        masked_batch = mask_tokenized_inputs(encoded_batch, mlm_fraction=config["mlm_fraction"], mask_token_id=tokenizer.mask_token_id).to(device)
        labels = torch.where(masked_batch.input_ids == tokenizer.mask_token_id, encoded_batch["input_ids"], -100)
        outputs = model(**masked_batch, labels=labels, output_hidden_states=False)
        loss = outputs.loss
        test_loss_logs = {"test/loss": loss.item()}
        if avg_test_logs is None:
            avg_test_logs = test_loss_logs.copy()
        else:
            for k in test_loss_logs:
                avg_test_logs[k] += test_loss_logs[k]
                
    for k in avg_test_logs:
        avg_test_logs[k] /= len(test_loader)
        
    for k, v in avg_test_logs.items():
        wandb.run.summary[k] = v


    print("Training complete")

        
        
    
if __name__=="__main__":
    ## ADD PARSER
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_number", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="devel_ESM2_finetuning")
    parser.add_argument("--experiment_group", type=str, default="FinetuneESM2")
    parser.add_argument("--project_folder", type=str, default="/home/gvisona/Projects/SelfPeptides")
    parser.add_argument("--data_folder", type=str, default="/home/gvisona/Projects/SelfPeptides")
    
    parser.add_argument("--pretrained_model", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--peptides_dataframes", type=str, nargs="+", default=[
        "processed_data/Immunogenicity/Processed_TCell_IEDB_beta_summed.csv",
        "processed_data/Immunogenicity/DHLAP_immunogenicity_data.csv",
        "processed_data/Binding_Affinity/DHLAP_binding_affinity_data.csv",
        "processed_data/Binding_Affinity/HLA_Ligand_Atlas_processed.csv"        
    ])
    
    parser.add_argument("--mlm_fraction", type=float, default=0.15)
    
    parser.add_argument("--max_updates", type=int, default=5)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--validate_every_n_updates", type=int, default=1)
    parser.add_argument("--accumulate_batches", type=int, default=1)
    
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.1)
    
    parser.add_argument("--batch_size", type=int, default=8)
    
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--nesterov_momentum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_frac", type=float, default=0.01)
    parser.add_argument("--ramp_up", type=float, default=0.3)
    parser.add_argument("--cool_down", type=float, default=0.6)
    
    
    args = parser.parse_args()
    config = vars(args)
    

    finetune_model(config=config)
    
    