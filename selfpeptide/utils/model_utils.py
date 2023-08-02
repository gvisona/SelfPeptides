import json
import torch
import os
from selfpeptide.model.binding_affinity_classifier import Peptide_HLA_BindingClassifier


def load_binding_model(folder, device="cpu"):
    with open(os.path.join(folder, "config.json"), "r") as f:
        config = json.load(f)
    model = Peptide_HLA_BindingClassifier(config, device)
    model.to(device)
    checkpoint_path = os.path.join(folder, "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model