import json
import torch
import os
from selfpeptide.model.binding_affinity_classifier import Peptide_HLA_BindingClassifier
from selfpeptide.model.peptide_embedder import SelfPeptideEmbedder_withProjHead


def load_binding_model(folder, device="cpu"):
    with open(os.path.join(folder, "config.json"), "r") as f:
        config = json.load(f)
    config["pretrained_aa_embeddings"] = "none"
    model = Peptide_HLA_BindingClassifier(config, device)
    model.to(device)
    checkpoint_path = os.path.join(folder, "checkpoints", "001_checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def load_sns_model(folder, device="cpu"):
    with open(os.path.join(folder, "config.json"), "r") as f:
        config = json.load(f)
    config["pretrained_aa_embeddings"] = "none"
    model = SelfPeptideEmbedder_withProjHead(config, device)
    model.to(device)
    checkpoint_path = os.path.join(folder, "checkpoints", "001_checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def build_immunogenicity_model():
    pass