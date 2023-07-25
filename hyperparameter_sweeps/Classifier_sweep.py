import sys
sys.path.insert(0, "..")
sys.path.insert(0, "/home/gvisona/Projects/Immunology")
sys.path.insert(0, "/home/gvisona/Immunology")
sys.path.insert(0, "/fast/gvisona/Immunology")
import os
import yaml
import pprint

from argparse import ArgumentParser
import wandb
from training_scripts.train_selfpeptide_classifier import train


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    sweep_id = '8pq7nowg'
    
    if os.path.exists("/home/gvisona/Projects/Immunology"):
        project_folder = "/home/gvisona/Projects/Immunology"
    else:
        project_folder = "/home/gvisona/Immunology"
    
    with open(os.path.join(project_folder, "hyperparameter_sweeps", "Classifier_config.yml"), "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    config = {
        "seed": args.seed,
        "experiment_name": sweep_config["parameters"]["experiment_name"]["value"]
    }
    
    def train_func():
        train(config)
    
    wandb.agent(sweep_id, function=train_func, count=1, entity="gvisona", project=config["experiment_name"])

    