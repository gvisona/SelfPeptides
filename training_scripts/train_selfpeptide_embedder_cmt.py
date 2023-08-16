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
from tqdm import tqdm
import math
import json

from selfpeptide.utils.data_utils import PreSplit_Self_NonSelf_PeptideDataset, split_pretokenized_data, PreTokenized_HumanPeptidesDataset
from selfpeptide.utils.training_utils import lr_schedule, warmup_constant_lr_schedule, eval_classification_metrics, CustomCMT_AllTriplets_Loss
from selfpeptide.model.peptide_embedder import SelfPeptideEmbedder_withProjHead

MAX_K = 11

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
    if run_name is None or len(run_name) < 1 or not config["wandb_sweep"]:
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
    # checkpoint_path = os.path.join(output_folder, "checkpoint.pt")
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
    
    val_set, test_set, ref_set, train_set = split_pretokenized_data(config["hdf5_dataset"],
                                                                    holdout_sizes=[
                                                                        config["val_size"], config["test_size"], config["ref_size"]],
                                                                    random_state=config["seed"], test_run=config["test_run"])

    val_dset = PreSplit_Self_NonSelf_PeptideDataset(*val_set)
    test_dset = PreSplit_Self_NonSelf_PeptideDataset(*test_set)
    ref_dset = PreSplit_Self_NonSelf_PeptideDataset(*ref_set)
    train_dset = PreSplit_Self_NonSelf_PeptideDataset(*train_set)

    train_loader = DataLoader(
        train_dset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_dset, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test_dset, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    ref_loader = DataLoader(
        ref_dset, batch_size=config["batch_size"], shuffle=False, drop_last=False)

    model = SelfPeptideEmbedder_withProjHead(config, device)
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

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config.get("momentum", 0.9),
                                nesterov=config.get(
                                    "nesterov_momentum", False),
                                weight_decay=config['weight_decay'])

    # def lr_lambda(s): return lr_schedule(s, min_frac=config['min_frac'], total_iters=config["max_updates"],
    #                                      ramp_up=config['ramp_up'], cool_down=config['cool_down'])
    if resume_checkpoint_path is None and config["init_checkpoint"] is None:
        def lr_lambda(s): return warmup_constant_lr_schedule(s, min_frac=config['min_frac'], total_iters=config["max_updates"],
                                         ramp_up=config['ramp_up'])
    else:
        lr_lambda = lambda s: 1.0
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    margin = config.get("margin", 0.8)
    loss_function = CustomCMT_AllTriplets_Loss(
        s=config.get("loss_s", 1.0), m=margin)

    best_val_metric = 0.0
    best_metric_iter = 0
    n_iters_per_val_cycle = config["accumulate_batches"] * \
        config["validate_every_n_updates"]
    perform_validation = False
    log_results = False
    avg_train_logs = None

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
            peptides, labels = train_batch
            if torch.is_tensor(peptides):
                peptides = peptides.to(device)
            labels = labels.to(device)

            projections, embeddings = model(peptides)
            loss, train_loss_logs = loss_function(projections, labels)

            if avg_train_logs is None:
                avg_train_logs = train_loss_logs.copy()
            else:
                for k in train_loss_logs:
                    avg_train_logs[k] += train_loss_logs[k]
            loss.backward()

            if n_iter % config['accumulate_batches'] == 0:
                # print("UPDATE")
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                n_update += 1
                pbar.update(1)
                if n_update % config["validate_every_n_updates"] == 0:
                    perform_validation = True

            if perform_validation:
                perform_validation = False
                model.eval()

                ref_proj = []
                ref_labels = []
                for ix, ref_batch in enumerate(ref_loader):
                    peptides, labels = ref_batch
                    if torch.is_tensor(peptides):
                        peptides = peptides.to(device)
                    projections, embeddings = model(peptides)
                    ref_proj.append(projections.detach())
                    ref_labels.append(labels.detach())
                ref_proj = torch.cat(ref_proj, dim=0)
                ref_labels = torch.cat(ref_labels)
                
                ref_proj = ref_proj / ref_proj.norm(dim=1)[:, None]
                ref_labels = ((ref_labels+1)/2).cpu()

                val_proj_idxs = []
                val_labels = []
                for ix, val_batch in enumerate(val_loader):
                    peptides, labels = val_batch
                    if torch.is_tensor(peptides):
                        peptides = peptides.to(device)
                    labels = labels.to(device)
                    projections, embeddings = model(peptides)
                    # val_proj.append(projections.detach())
                    val_proj = projections.detach()
                    val_proj = val_proj / val_proj.norm(dim=1)[:, None]
                    proj_similarity = torch.mm(
                        val_proj, ref_proj.transpose(0, 1)).cpu()
                    
                    vals, proj_idxs = torch.topk(proj_similarity, MAX_K, dim=1)
                    val_proj_idxs.append(proj_idxs.detach())
                    val_labels.append(labels.detach())

                val_proj_idxs = torch.cat(val_proj_idxs, dim=0)
                val_labels = torch.cat(val_labels)
                val_labels = ((val_labels+1)/2).cpu()
        
                val_classification_metrics = {}


                for K in [11]:
                    proj_k_idxs = val_proj_idxs[:, :K]
                    proj_knn_classes = ref_labels[proj_k_idxs]
                    pred_proj_labels = torch.mean(proj_knn_classes, dim=1)
                    k_mean_classification_metrics = eval_classification_metrics(val_labels, pred_proj_labels,
                                                                                is_logit=False,
                                                                                threshold=0.5)
                    k_mean_classification_metrics = {
                        "val_proj_KNN/K_{}_mean/".format(K)+k: v for k, v in k_mean_classification_metrics.items()}
                    val_classification_metrics.update(
                        k_mean_classification_metrics)

                epoch_val_metric = val_classification_metrics["val_proj_KNN/K_11_mean/MCC"]
                
                if epoch_val_metric > best_val_metric:
                    best_val_metric = epoch_val_metric
                    best_metric_iter = n_iter
                    torch.save(model.state_dict(), checkpoint_path)
                    resume_config = {"n_iter": n_iter, "n_update": n_update}
                    with open(os.path.join(checkpoints_folder, checkpoint_label+".json"), "w") as f:
                        json.dump(resume_config, f, indent=2)

                log_results = True
                model.train()

            if log_results:
                log_results = False
                for k in avg_train_logs:
                    avg_train_logs[k] /= (config['accumulate_batches']
                                          * config["validate_every_n_updates"])

                current_lr = scheduler.get_last_lr()[0]
                logs = {"learning_rate": current_lr,
                        "accumulated_batch": n_update}

                logs.update(avg_train_logs)
                logs.update(val_classification_metrics)
                wandb.log(logs)
                avg_train_logs = None

            if config.get("early_stopping", False):
                if (n_iter - best_metric_iter)/n_iters_per_val_cycle > config['patience']:
                    print("Val metric not improving, stopping training..\n\n")
                    break

    pbar.close()

    print(os.path.exists(checkpoint_path))
    print(f"\nLoading state dict from {checkpoint_path}\n\n")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # print("Testing model..")
    # test_proj = []
    # test_labels = []
    # for ix, test_batch in enumerate(test_loader):
    #     peptides, labels = test_batch
    #     if torch.is_tensor(peptides):
    #         peptides = peptides.to(device)
    #     labels = labels.to(device)
    #     projections, embeddings = model(peptides)
    #     test_proj.append(projections.detach())
    #     test_labels.append(labels.detach())

    # test_proj = torch.cat(test_proj, dim=0)
    # test_labels = torch.cat(test_labels)

    # ref_proj = []
    # ref_labels = []
    # for ix, ref_batch in enumerate(ref_loader):
    #     peptides, labels = ref_batch
    #     if torch.is_tensor(peptides):
    #         peptides = peptides.to(device)
    #     projections, embeddings = model(peptides)
    #     ref_proj.append(projections.detach())
    #     ref_labels.append(labels.detach())
    # ref_proj = torch.cat(ref_proj, dim=0)
    # ref_labels = torch.cat(ref_labels)

    # test_proj = test_proj / test_proj.norm(dim=1)[:, None]
    # ref_proj = ref_proj / ref_proj.norm(dim=1)[:, None]
    # similarity = torch.mm(test_proj, ref_proj.transpose(0, 1)).cpu()

    # test_classification_metrics = {}

    # test_labels = ((test_labels+1)/2).cpu()
    # ref_labels = ((ref_labels+1)/2).cpu()

    # vals, idxs = torch.topk(similarity, MAX_K, dim=1)
    # for K in [11]:
    #     k_idxs = idxs[:, :K]
    #     proj_k_distances = 1.0 - vals[:, :K]
    #     # knn_weights = (proj_k_distances[:, -1:] - proj_k_distances)/(
    #     #     torch.clamp(proj_k_distances[:, -1] - proj_k_distances[:, 0], min=1e-8)[:, None])
    #     # knn_weights = knn_weights / knn_weights.sum(dim=1)[:, None]
    #     knn_classes = ref_labels[k_idxs]
    #     test_pred_labels = torch.mean(knn_classes, dim=1)
    #     # test_weighted_proj_labels = (
    #     #                 knn_weights * knn_classes).sum(dim=1)
    #     k_mean_classification_metrics = eval_classification_metrics(test_labels, test_pred_labels,
    #                                                                 is_logit=False,
    #                                                                 threshold=0.5)
    #     k_mean_classification_metrics = {
    #         "K_{}_mean/".format(K)+k: v for k, v in k_mean_classification_metrics.items()}

    #     test_classification_metrics.update(k_mean_classification_metrics)
        
    #     # weighted_k_mean_classification_metrics = eval_classification_metrics(test_labels, test_weighted_proj_labels,
    #     #                                                             is_logit=False,
    #     #                                                             threshold=0.5)
    #     # weighted_k_mean_classification_metrics = {
    #     #     "weighted_K_{}_mean/".format(K)+k: v for k, v in weighted_k_mean_classification_metrics.items()}

    #     # test_classification_metrics.update(weighted_k_mean_classification_metrics)

    # test_classification_metrics = {
    #     "test_KNN_class/"+k: v for k, v in test_classification_metrics.items()}
    # for k, v in test_classification_metrics.items():
    #     wandb.run.summary[k] = v

    # print("Evaluating Cosine Centroid for Human Peptides")
    # p_dset = PreTokenized_HumanPeptidesDataset(
    #     config["hdf5_dataset"], test_run=config["test_run"])
    # p_loader = DataLoader(
    #     p_dset, batch_size=config["batch_size"], drop_last=False)

    # ref_human_peptides_vector = None
    # n_peptides = len(p_dset)
    # for ix, peptides in tqdm(enumerate(p_loader)):
    #     if torch.is_tensor(peptides):
    #         peptides = peptides.to(device)
    #     projections, embeddings = model(peptides)

    #     projections = projections / projections.norm(dim=1)[:, None]
    #     if ref_human_peptides_vector is None:
    #         ref_human_peptides_vector = torch.sum(projections.detach(), dim=0)
    #     else:
    #         ref_human_peptides_vector += torch.sum(projections.detach(), dim=0)
    # ref_human_peptides_vector /= n_peptides
    # ref_human_peptides_vector = (ref_human_peptides_vector / 
    #                             ref_human_peptides_vector.norm())
    # model.human_peptides_cosine_centroid = ref_human_peptides_vector
    # torch.save(model.state_dict(), checkpoint_path)

    print("Training complete!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="devel")
    parser.add_argument("--experiment_group", type=str, default="cmt_embedder")
    parser.add_argument("--project_folder", type=str,
                        default="/home/gvisona/Projects/SelfPeptides")

    parser.add_argument("--hdf5_dataset", type=str,
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/Self_nonSelf/pre_tokenized_peptides_dataset.hdf5")
    parser.add_argument("--pretrained_aa_embeddings", type=str,
                        default="/home/gvisona/Projects/SelfPeptides/processed_data/aa_embeddings/normalized_learned_BA_AA_embeddings.npy")
    parser.add_argument("--init_checkpoint", default=None)

    parser.add_argument("--max_updates", type=int, default=100)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--validate_every_n_updates", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulate_batches", type=int, default=4)

    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--test_size", type=int, default=256)
    parser.add_argument("--ref_size", type=int, default=256)

    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction)
    parser.add_argument("--test_run", action=argparse.BooleanOptionalAction)

    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov_momentum", action=argparse.BooleanOptionalAction)
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
    parser.add_argument("--PMA_ln", action=argparse.BooleanOptionalAction)
    
    parser.add_argument("--wandb_sweep", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    config = vars(args)
    # config["test_run"] = True
    # config["init_checkpoint"] = "/home/gvisona/Projects/SelfPeptides/outputs/cmt_embedder/devel/lively-breeze-89/checkpoints/001_checkpoint.pt"

    train(config=config)
