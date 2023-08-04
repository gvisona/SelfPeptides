import pandas as pd
import numpy as np
from argparse import ArgumentParser 
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from os.path import exists, join
from tqdm import tqdm
import math


from selfpeptide.utils.data_utils import PeptideTripletsDataset, Self_NonSelf_PeptideDataset
from selfpeptide.utils.training_utils import lr_schedule, eval_classification_metrics, CustomCMT_AllTriplets_Loss
from selfpeptide.model.peptide_embedder import SelfPeptideEmbedder_withProjHead



    

def train(config=None, init_wandb=True):
    if init_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=config['experiment_name'],
            
            # track hyperparameters and run metadata
            config=config
        )
    
        config = wandb.config
    
    run_name = wandb.run.name
    if run_name is None or len(run_name)<1:
        run_name = str(config["seed"])
        
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    if not exists(config['project_folder']):
        raise ValueError("Project folder does not exist")
    pass

    output_folder = join(config['project_folder'], 
                         "outputs", 
                         config['experiment_group'], 
                         config['experiment_name'], 
                         run_name)
    os.makedirs(output_folder, exist_ok=True)
    checkpoint_path = os.path.join(output_folder, "checkpoint.pt")

    
    val_dset = Self_NonSelf_PeptideDataset(config["hdf5_dataset"], gen_size=config["val_size"], )
    ref_dset = Self_NonSelf_PeptideDataset(config["hdf5_dataset"], gen_size=config["ref_size"], val_size=config["val_size"])
    train_dset = Self_NonSelf_PeptideDataset(config["hdf5_dataset"], gen_size=config["gen_size"], 
                                             val_size=config["val_size"]+config["ref_size"], test_run=config["test_run"])
    
    train_loader = DataLoader(train_dset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    # val_loader = DataLoader(val_dset, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    # ref_loader = DataLoader(ref_dset, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    
    
    
    model = SelfPeptideEmbedder_withProjHead(config, device)
    model.to(device)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    
    with open(join(output_folder, "architecture.txt"), "w") as f:
        f.write(model.__repr__())
    
    max_iters = config["max_updates"] * config["accumulate_batches"] + 1
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config.get("momentum", 0.9),
                            nesterov=config.get("nesterov_momentum", False),
                            weight_decay=config['weight_decay'])
    # lr_lambda = lambda s: lr_schedule(s, min_frac=config['min_frac'], total_iters=config["max_updates"], 
    #                                   ramp_up=config['ramp_up'], cool_down=config['cool_down'])
    lr_lambda = lambda s: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    
    margin = config.get("margin", 0.8)
    loss_function = CustomCMT_AllTriplets_Loss(s=config.get("loss_s", 1.0), m=margin)
    
    gen_train = iter(train_loader)    
    best_val_metric = 0.0
    best_metric_iter = 0
    n_iters_per_val_cycle = config["accumulate_batches"] * config["validate_every_n_updates"]    
    n_update = -1
    perform_validation = False
    log_results = False
    avg_train_logs = None

    # wandb.watch(model, log='gradients', log_freq=config["validate_every_n_updates"])
    
    for n_iter in tqdm(range(max_iters)):
        try:
            train_batch = next(gen_train)
        except StopIteration:
            train_dset.refresh_data()
            gen_train = iter(train_loader)
            train_batch = next(gen_train)
            
        peptides, labels = train_batch
        if torch.is_tensor(peptides):
            peptides = peptides.to(device)
        labels = labels.to(device)
            
        
        projections, embeddings = model(peptides)
        loss, train_loss_logs = loss_function(projections, labels)
        
        
        # train_loss_logs = {"train/CMT_loss": loss.item()}
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
            perform_validation = False
            # model.eval()
            
            log_results = True       
            # model.train()
            
        if log_results:
            log_results = False
            for k in avg_train_logs:
                avg_train_logs[k] /= (config['accumulate_batches'] * config["validate_every_n_updates"])
            
            
            current_lr = scheduler.get_last_lr()[0]
            # val_train_difference = avg_val_logs["val/triplet_loss"] - avg_train_logs["train/triplet_loss"]
            logs = {"learning_rate": current_lr, 
                    "accumulated_batch": n_update}
                    # "val_train_difference": val_train_difference}
            
            logs.update(avg_train_logs)
            # logs.update(best_metric_iter)
            # logs.update(val_classification_metrics)
            wandb.log(logs)      
            avg_train_logs = None
            
        # if config.get("early_stopping", False):
        #     if (n_iter - best_metric_iter)/n_iters_per_val_cycle>config['patience']:
        #         print("Val metric not improving, stopping training..\n\n")
        #         break
            
            
    print("Training complete!")









if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="gradient_check")
    parser.add_argument("--experiment_group", type=str, default="triplet_loss_embedder")
    parser.add_argument("--project_folder", type=str, default="/home/gvisona/Projects/SelfPeptides")
    
    parser.add_argument("--hdf5_dataset", type=str, default="/home/gvisona/Projects/SelfPeptides/processed_data/Self_nonSelf/pre_tokenized_peptides_dataset.hdf5")
    parser.add_argument("--pretrained_aa_embeddings", type=str, 
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/aa_embeddings/normalized_learned_BA_AA_embeddings.npy")
    
    

    parser.add_argument("--max_updates", type=int, default=100000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--validate_every_n_updates", type=int, default=50)
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulate_batches", type=int, default=4)
    
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--ref_size", type=int, default=50)
    parser.add_argument("--gen_size", type=int, default=1000)
    
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--test_run", type=bool, default=True)
    
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov_momentum", action="store_true", default=True)
    parser.add_argument("--min_frac", type=float, default=0.01)
    parser.add_argument("--ramp_up", type=float, default=0.3)
    parser.add_argument("--cool_down", type=float, default=0.6)


    parser.add_argument("--dropout_p", type=float, default=0.15)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--projection_hidden_dim", type=int, default=2048)
    parser.add_argument("--projection_dim", type=int, default=32)
    parser.add_argument("--transf_hidden_dim", type=int, default=2048)
    parser.add_argument("--n_attention_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--PMA_num_heads", type=int, default=1)
    
    args = parser.parse_args()
    
    config = vars(args)

    train(config=config)