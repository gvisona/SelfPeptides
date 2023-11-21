import torch
import torch.nn as nn

# from model.encoders.MSA_transformer_encoder import *
from selfpeptide.model.peptide_embedder import PeptideEmbedder
from selfpeptide.model.hla_embedder import Joint_HLA_Embedder
from selfpeptide.model.components import ResMLP_Network
    
    
    
    

class Peptide_HLA_BindingClassifier(nn.Module):
    def __init__(self, config, device='cpu'):
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
        
    def forward(self, peptides, hla_pseudoseqs, hla_proteins, *args):
        peptide_embs = self.peptide_embedder(peptides)
        hla_embs = self.hla_embedder(hla_pseudoseqs, hla_proteins)
        embs = torch.cat([peptide_embs, hla_embs], dim=-1)
        joint_embeddings = self.joint_embedder(embs)
        output = self.classifier_model(joint_embeddings)
        return output, [joint_embeddings, peptide_embs, hla_embs]