import torch
import torch.nn as nn
from selfpeptide.model.components import ResMLP_Network
from selfpeptide.model.peptide_embedder import PeptideEmbedder

    
    
class ImmunogenicityClassifier(nn.Module):
    def __init__(self, config, device, binding_model=None, sns_model=None, epsilon=1e-3):
        super().__init__()
        self.config = config
        self.device = device
        self.epsilon = epsilon
        
        self.binding_model = binding_model 
        self.sns_model = sns_model
        
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
        
        self.beta_regression_output_module.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.01)
            module.weight.data.normal_(mean=1.0, std=0.01)
            
    def forward(self, peptides, hlas, *args):
        binding_score, (binding_peptides_embs, binding_hlas_embs) = self.binding_model(peptides, hlas)
        sns_peptides_projections, sns_peptides_embs, sns_scores = self.sns_model(peptides, return_sns_score=True)
        
        binding_score = torch.sigmoid(binding_score)
        peptide_imm_embs = self.immunogenicity_aa_embedder(peptides)
        hla_imm_embs = self.immunogenicity_aa_embedder(hlas)
        mlp_input = torch.cat([binding_peptides_embs, binding_hlas_embs, 
                               sns_peptides_embs, peptide_imm_embs, 
                               hla_imm_embs], dim=1)
        
        mlp_output = self.joint_mlp(mlp_input)
        beta_regr_input = torch.cat([binding_score, sns_scores, mlp_output], dim=1)
        beta_output = self.beta_regression_output_module(beta_regr_input)
        X_out = torch.zeros_like(beta_output, device=self.device)
        beta_mean = self.epsilon + (1-2*self.epsilon) * beta_output[:, 0]
        X_out[:, 0] = beta_mean
        X_out[:, 1] = self.epsilon + (1-2*self.epsilon) * (beta_output[:, 1]/3) * beta_mean * (1-beta_mean)
        return X_out