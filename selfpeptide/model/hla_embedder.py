import json
import torch
import numpy as np
from typing import List, Dict, Union, Any

import torch.nn as nn

class Pretrained_HLA_Embedder(nn.Module):
    """
    A module that embeds HLA sequences using pre-trained embeddings.
    """
    def __init__(self, config: Dict[str, Any], device: str = "cpu", prefix: str = "") -> None:
        """
        Initializes the Pretrained_HLA_Embedder module.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
            device (str): The device to use for computation (default: "cpu").
            prefix (str): The prefix to use for loading configuration parameters (default: "").

        """
        super().__init__()
        with open(config[prefix+"seq2idx_file"], "r") as f:
            self.seq2idx = json.load(f)
        self.idx2seq = {i:s for s, i in self.seq2idx.items()}
        self.device = device
        embeddings = np.load(config[prefix+"embeddings_file"])
        self.pretrained_embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings))
        self.pretrained_embeddings.to(device)
    
    def forward(self, hla_seqs: List[str]) -> torch.Tensor:
        """
        Forward pass of the Pretrained_HLA_Embedder module.

        Args:
            hla_seqs (List[str]): A list of HLA sequences to embed.

        Returns:
            torch.Tensor: The embedded representations of the input HLA sequences.

        """
        idxs = [self.seq2idx[s] for s in hla_seqs]
        idxs_tensor = torch.LongTensor(idxs).to(self.device)
        embs = self.pretrained_embeddings(idxs_tensor)
        return embs
    
    
class Joint_HLA_Embedder(nn.Module):
    """
    A module that embeds joint HLA pseudosequences and protein sequences.
    """
    def __init__(self, config: Dict[str, Any], device: str = "cpu") -> None:
        """
        Initializes the Joint_HLA_Embedder module.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
            device (str): The device to use for computation (default: "cpu").

        """
        super().__init__()
        self.pseudoseq_embedder = Pretrained_HLA_Embedder(config, device=device, prefix="pseudoseq_")
        self.protein_embedder = Pretrained_HLA_Embedder(config, device=device, prefix="protein_")
        
        self.projection_layer = nn.Linear(2*config["hla_embedding_dim"], config["embedding_dim"]).to(device)
        
    def forward(self, hla_pseudoseqs: List[str], hla_protein_seqs: List[str]) -> torch.Tensor:
        """
        Forward pass of the Joint_HLA_Embedder module.

        Args:
            hla_pseudoseqs (List[str]): A list of HLA pseudosequences to embed.
            hla_protein_seqs (List[str]): A list of HLA protein sequences to embed.

        Returns:
            torch.Tensor: The embedded representations of the input joint HLA sequences.

        """
        pseq_embs = self.pseudoseq_embedder(hla_pseudoseqs)
        prot_embs = self.protein_embedder(hla_protein_seqs)
        joint_embs = torch.cat([pseq_embs, prot_embs], dim=1)
        return self.projection_layer(joint_embs)