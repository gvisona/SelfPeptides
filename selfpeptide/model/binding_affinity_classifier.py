import torch
import torch.nn as nn

# from model.encoders.MSA_transformer_encoder import *
from selfpeptide.model.peptide_embedder import PeptideEmbedder
from selfpeptide.model.components import ResMLP_Network
    

class Peptide_HLA_BindingClassifier(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        self.aa_sequence_embedder = PeptideEmbedder(config, device)
        self.classifier_model = ResMLP_Network(config, device)
        
    def forward(self, peptides, hlas, *args):
        peptide_embs = self.aa_sequence_embedder(peptides)
        hla_embs = self.aa_sequence_embedder(hlas)
        embs = [peptide_embs, hla_embs]
        output = self.classifier_model(*embs)
        return output, [peptide_embs, hla_embs]