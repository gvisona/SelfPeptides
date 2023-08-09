import torch
import torch.nn as nn
from selfpeptide.model.components import ResMLP_Network
from selfpeptide.model.peptide_embedder import PeptideEmbedder

class ImmunogenicityClassifier(nn.Module):
    def __init__(self, config, device, binding_model=None, sns_model=None, human_sns_vector=None, epsilon=1e-8):
        self.config = config
        self.device = device
        self.epsilon = epsilon
        
        self.binding_model = binding_model 
        self.sns_model = sns_model
        
        self.human_sns_vector = torch.tensor(human_sns_vector, requires_grad=False) if human_sns_vector is not None else None
        self.cos_sim = nn.CosineSimilarity()
        
        
        # Freeze binding and SnS model
        for p in self.binding_model.parameters():
            p.requires_grad = False
        for p in self.sns_model.parameters():
            p.requires_grad = False
            
        self.immunogenicity_aa_embedder = PeptideEmbedder(config, device)
        self.joint_mlp = ResMLP_Network(config, device)
        
        self.beta_regression_output_module = nn.Sequential(nn.Linear(config["output_dim"]+2, config["beta_regr_hidden_dim"]), 
                                                           nn.ReLU(),
                                                           nn.Linear(config["beta_regr_hidden_dim"], 2),
                                                           nn.Sigmoid())
            
    def forward(self, peptides, hlas, *args):
        binding_score, (binding_peptides_embs, binding_hlas_embs) = self.binding_model(peptides, hlas)
        sns_peptides_projections, sns_peptides_embs = self.sns_model(peptides)
        
        human_sns_score = self.cos_sim(sns_peptides_embs, self.human_sns_vector)
        
        
        peptide_imm_embs = self.immunogenicity_aa_embedder(peptides)
        hla_imm_embs = self.immunogenicity_aa_embedder(hlas)
        mlp_input = torch.cat([binding_peptides_embs, binding_hlas_embs, 
                               sns_peptides_embs, peptide_imm_embs, 
                               hla_imm_embs], dim=1)
        
        mlp_output = self.joint_mlp(mlp_input)
        beta_regr_input = torch.cat([binding_score, human_sns_score, mlp_output], dim=1)
        beta_output = self.beta_regression_output_module(beta_regr_input)
        X_out = torch.zeros_like(beta_output, device=self.device)
        X_out[:, 0] = self.epsilon + (1-2*self.epsilon) * beta_output[:, 0]
        X_out[:, 1] = self.epsilon + (1-2*self.epsilon) * (beta_output[:, 1]/3) * X_out[:, 0] * (1-X_out[:, 0])
        return X_out