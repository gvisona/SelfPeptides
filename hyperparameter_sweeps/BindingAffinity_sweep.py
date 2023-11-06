import sys
sys.path.insert(0, "..")
sys.path.insert(0, "/home/gvisona/Projects/SelfPeptides")
sys.path.insert(0, "/home/gvisona/SelfPeptides")
sys.path.insert(0, "/fast/gvisona/SelfPeptides")
import os
import yaml
import pprint

from argparse import ArgumentParser
import wandb
from training_scripts.train_binding_model import train


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    sweep_id = 'twn67fbv'
    
    if os.path.exists("/home/gvisona/Projects/SelfPeptides"):
        project_folder = "/home/gvisona/Projects/SelfPeptides"
    else:
        project_folder = "/home/gvisona/SelfPeptides"
    
    with open(os.path.join(project_folder, "hyperparameter_sweeps", "BindingAffinity_config.yml"), "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        
    config = {
        "seed": args.seed,
        "run_number": args.seed,
        "experiment_name": sweep_config["parameters"]["experiment_name"]["value"]
    }
    
    def train_func():
        train(config)
    
    wandb.agent(sweep_id, function=train_func, count=1, entity="gvisona", project=config["experiment_name"])

    