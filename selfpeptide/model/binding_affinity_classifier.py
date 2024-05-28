import torch
import torch
from selfpeptide.model.peptide_embedder import PeptideEmbedder
from selfpeptide.model.hla_embedder import Joint_HLA_Embedder
from selfpeptide.model.components import ResMLP_Network
from typing import List, Tuple

import torch.nn as nn

# from model.encoders.MSA_transformer_encoder import *

class Peptide_HLA_BindingClassifier(nn.Module):
    """
    A class representing a peptide-HLA binding classifier.

    Args:
        config (dict): A dictionary containing configuration parameters.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.

    Attributes:
        device (str): The device the model is running on.
        peptide_embedder (PeptideEmbedder): The peptide embedder module.
        hla_embedder (Joint_HLA_Embedder): The HLA embedder module.
        joint_embedder (nn.Sequential): The joint embedder module.
        classifier_model (ResMLP_Network): The classifier model.

    Methods:
        forward: Performs a forward pass through the model.

    """

    def __init__(self, config: dict, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.peptide_embedder = PeptideEmbedder(config, device)
        self.hla_embedder = Joint_HLA_Embedder(config, device)
        
        self.joint_embedder = nn.Sequential(
            nn.Linear(2*config["embedding_dim"], config["joint_embedder_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["joint_embedder_hidden_dim"], config["embedding_dim"])
        )
        
        self.classifier_model = ResMLP_Network(config, device)
        
    def forward(self, peptides: torch.Tensor, hla_pseudoseqs: torch.Tensor, hla_proteins: torch.Tensor, *args) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Performs a forward pass through the model.

        Args:
            peptides (torch.Tensor): The input peptide sequences.
            hla_pseudoseqs (torch.Tensor): The input HLA pseudosequences.
            hla_proteins (torch.Tensor): The input HLA proteins.
            *args: Additional arguments.

        Returns:
            output (torch.Tensor): The output of the classifier model.
            embeddings (list): A list of embeddings used in the forward pass.

        """
        peptide_embs = self.peptide_embedder(peptides)
        hla_embs = self.hla_embedder(hla_pseudoseqs, hla_proteins)
        embs = torch.cat([peptide_embs, hla_embs], dim=-1)
        joint_embeddings = self.joint_embedder(embs)
        output = self.classifier_model(joint_embeddings)
        return output, [joint_embeddings, peptide_embs, hla_embs]
