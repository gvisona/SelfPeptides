import torch
import torch.nn as nn
from selfpeptide.model.peptide_embedder import PeptideEmbedder, SelfPeptideEmbedder_withProjHead
from selfpeptide.model.binding_affinity_classifier import Peptide_HLA_BindingClassifier
from selfpeptide.model.components import ResMLP_Network
from selfpeptide.model.components import FDS
import warnings
import json

# class ImmunogenicityClassifier(nn.Module):
#     def __init__(self, config, device, binding_model=None, sns_model=None, epsilon=1e-3):
#         super().__init__()
#         self.config = config
#         self.device = device
#         self.epsilon = epsilon
        
#         self.binding_model = binding_model 
#         self.sns_model = sns_model
        
#         # Freeze binding and SnS model
#         for p in self.binding_model.parameters():
#             p.requires_grad = False
#         for p in self.sns_model.parameters():
#             p.requires_grad = False
            
#         self.immunogenicity_aa_embedder = PeptideEmbedder(config, device)
#         self.joint_mlp = ResMLP_Network(config, device)
        
#         self.beta_regression_output_module = nn.Sequential(nn.Linear(config["output_dim"]+2, config["beta_regr_hidden_dim"]), 
#                                                            nn.ReLU(),
#                                                            nn.Linear(config["beta_regr_hidden_dim"], 2),
#                                                            nn.Sigmoid())
        
#         self.beta_regression_output_module.apply(self._init_weights)
        
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
#             if module.bias is not None:
#                 module.bias.data.normal_(mean=0.0, std=0.01)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.normal_(mean=0.0, std=0.01)
#             module.weight.data.normal_(mean=1.0, std=0.01)
            
#     def forward(self, peptides, hlas, *args):
#         binding_score, (binding_peptides_embs, binding_hlas_embs) = self.binding_model(peptides, hlas)
#         sns_peptides_projections, sns_peptides_embs, sns_scores = self.sns_model(peptides, return_sns_score=True)
        
#         binding_score = torch.sigmoid(binding_score)
#         peptide_imm_embs = self.immunogenicity_aa_embedder(peptides)
#         hla_imm_embs = self.immunogenicity_aa_embedder(hlas)
#         mlp_input = torch.cat([binding_peptides_embs, binding_hlas_embs, 
#                                sns_peptides_embs, peptide_imm_embs, 
#                                hla_imm_embs], dim=1)
        
#         mlp_output = self.joint_mlp(mlp_input)
#         beta_regr_input = torch.cat([binding_score, sns_scores, mlp_output], dim=1)
#         beta_output = self.beta_regression_output_module(beta_regr_input)
#         X_out = torch.zeros_like(beta_output, device=self.device)
#         beta_mean = self.epsilon + (1-2*self.epsilon) * beta_output[:, 0]
#         X_out[:, 0] = beta_mean
#         X_out[:, 1] = self.epsilon + (1-2*self.epsilon) * (beta_output[:, 1]/3) * beta_mean * (1-beta_mean)
#         return X_out
    

class ImmunogenicityClassifier(nn.Module):
    def __init__(self, config, device, epsilon=1e-3):
        super().__init__()
        self.config = config
        self.device = device
        self.epsilon = epsilon
        
        
            
        self.immunogenicity_aa_embedder = PeptideEmbedder(config, device)
        self.joint_mlp = ResMLP_Network(config, device)
        
        self.beta_regression_output_module = nn.Sequential(nn.Linear(config["output_dim"]+2, config["beta_regr_hidden_dim"]), 
                                                           nn.ReLU(),
                                                           nn.Linear(config["beta_regr_hidden_dim"], 2),
                                                           nn.Sigmoid())
            
    def forward(self, peptides, hlas, binding_peptides_embs, binding_hlas_embs, sns_peptides_embs, binding_score, sns_scores, *args):
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
    
    



    
class JointPeptidesNetwork(nn.Module):
    def __init__(self, imm_config, binding_config, sns_config, binding_checkpoint=None, sns_checkpoint=None, device="cpu"):
        super().__init__()
        if not isinstance(binding_config, dict):
            with open(binding_config, "r") as f:
                binding_config = json.load(f)
        binding_config["pretrained_aa_embeddings"] = "none"
        
        if not isinstance(sns_config, dict):
            with open(sns_config, "r") as f:
                sns_config = json.load(f)
        sns_config["pretrained_aa_embeddings"] = "none"
        
        self.binding_model = Peptide_HLA_BindingClassifier(binding_config, device=device) 
        if binding_checkpoint is not None:
            self.binding_model.load_state_dict(torch.load(binding_checkpoint, map_location=device))
        else:
            warnings.warn("Binding model not initialized")
        self.binding_model.eval()
        
        self.sns_model = SelfPeptideEmbedder_withProjHead(sns_config, device=device)
        if sns_checkpoint is not None:
            self.sns_model.load_state_dict(torch.load(sns_checkpoint, map_location=device))
        else:
            warnings.warn("SnS model not initialized")
        self.sns_model.eval()
        
        # Freeze binding and SnS model
        for p in self.binding_model.parameters():
            p.requires_grad = False
        for p in self.sns_model.parameters():
            p.requires_grad = False
            
        self.immunogenicity_model = ImmunogenicityClassifier(imm_config, device=device)
        self.immunogenicity_model.train()
    
    def forward(self, peptides, hlas, *args):
        binding_score, (binding_peptides_embs, binding_hlas_embs) = self.binding_model(peptides, hlas)
        sns_peptides_projections, sns_peptides_embs, sns_scores = self.sns_model(peptides, return_sns_score=True)
        
        output = self.immunogenicity_model(peptides, hlas, binding_peptides_embs, binding_hlas_embs, 
                                           sns_peptides_embs, binding_score, sns_scores)
        return output
    
    
    

    
class ImmunogenicityBinaryClassifier(nn.Module):
    def __init__(self, config, device, epsilon=1e-3):
        super().__init__()
        self.config = config
        self.device = device
        self.epsilon = epsilon
        
            
        self.immunogenicity_aa_embedder = PeptideEmbedder(config, device)
        self.joint_mlp = ResMLP_Network(config, device)
        

    def forward(self, peptides, hlas, binding_peptides_embs, binding_hlas_embs, sns_peptides_embs, binding_score, sns_scores, *args):
        peptide_imm_embs = self.immunogenicity_aa_embedder(peptides)
        hla_imm_embs = self.immunogenicity_aa_embedder(hlas)
        mlp_input = torch.cat([binding_peptides_embs, binding_hlas_embs, 
                               sns_peptides_embs, peptide_imm_embs, 
                               hla_imm_embs], dim=1)
        
        mlp_output = self.joint_mlp(mlp_input)
        return mlp_output
    
    
    
    
    
# class JointPeptidesNetwork_Classifier(nn.Module):
#     def __init__(self, imm_config, binding_config, sns_config, binding_checkpoint=None, sns_checkpoint=None, device="cpu"):
#         super().__init__()
#         if not isinstance(binding_config, dict):
#             with open(binding_config, "r") as f:
#                 binding_config = json.load(f)
#         binding_config["pretrained_aa_embeddings"] = "none"
        
#         if not isinstance(sns_config, dict):
#             with open(sns_config, "r") as f:
#                 sns_config = json.load(f)
#         sns_config["pretrained_aa_embeddings"] = "none"
        
#         self.binding_model = Peptide_HLA_BindingClassifier(binding_config, device=device) 
#         if binding_checkpoint is not None:
#             self.binding_model.load_state_dict(torch.load(binding_checkpoint, map_location=device))
#         else:
#             warnings.warn("Binding model not initialized")
#         self.binding_model.eval()
        
#         self.sns_model = SelfPeptideEmbedder_withProjHead(sns_config, device=device)
#         if sns_checkpoint is not None:
#             self.sns_model.load_state_dict(torch.load(sns_checkpoint, map_location=device))
#         else:
#             warnings.warn("SnS model not initialized")
#         self.sns_model.eval()
        
#         # Freeze binding and SnS model
#         for p in self.binding_model.parameters():
#             p.requires_grad = False
#         for p in self.sns_model.parameters():
#             p.requires_grad = False
            
#         self.immunogenicity_model = ImmunogenicityBinaryClassifier(imm_config, device=device)
#         self.immunogenicity_model.train()
    
#     def forward(self, peptides, hlas, *args):
#         binding_score, (binding_peptides_embs, binding_hlas_embs) = self.binding_model(peptides, hlas)
#         sns_peptides_projections, sns_peptides_embs, sns_scores = self.sns_model(peptides, return_sns_score=True)
        
#         output = self.immunogenicity_model(peptides, hlas, binding_peptides_embs, binding_hlas_embs, 
#                                            sns_peptides_embs, binding_score, sns_scores)
#         return output

class ImmunogenicityBetaModel(nn.Module):
    def __init__(self, config, device="cpu", epsilon=1e-3):
        super().__init__()
        self.config = config
        self.device = device
        self.epsilon = epsilon
        # assert config["output_dim"]==2, "Beta model requires 2 outputs"
        
        self.joint_mlp = ResMLP_Network(config, device)  
        
        self.output_module = nn.Sequential(
            nn.Linear(config["mlp_output_dim"], config["imm_regression_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["imm_regression_hidden_dim"], 2)
        )      
        self.use_FDS = config.get("use_FDS", False)
        if self.use_FDS:
            self.fds = FDS(device=device, **config).to(self.device)
        else:
            self.fds = None
        
        

    def forward(self, binding_score_logit, binding_peptides_embs, binding_hlas_embs, *args, target_labels=None):
        binding_score = torch.sigmoid(binding_score_logit).view(-1)
        mlp_input = torch.cat([binding_score.view(-1,1), binding_peptides_embs, binding_hlas_embs], dim=1)
        
        joint_embs = self.joint_mlp(mlp_input)
        classifier_input = joint_embs.clone()
        if target_labels is not None:
            classifier_input = self.fds.smooth(joint_embs, target_labels)
        output = self.output_module(joint_embs)
        
        means = self.epsilon + (1-2*self.epsilon) * torch.sigmoid(output[:, 0])
        posterior_means = binding_score * means
        precisions = 2 + torch.exp(output[:, 1])
        
        return torch.hstack([means.view(-1,1), posterior_means.view(-1,1), precisions.view(-1,1)]), joint_embs
    
    
class JointPeptidesNetwork_Beta(nn.Module):
    def __init__(self, imm_config, binding_config, binding_checkpoint=None, device="cpu"):
        super().__init__()
        if not isinstance(binding_config, dict):
            with open(binding_config, "r") as f:
                binding_config = json.load(f)
        binding_config["pretrained_aa_embeddings"] = "none"
        
        self.binding_model = Peptide_HLA_BindingClassifier(binding_config, device=device) 
        if binding_checkpoint is not None:
            self.binding_model.load_state_dict(torch.load(binding_checkpoint, map_location=device))
        else:
            warnings.warn("Binding model not initialized")
        self.binding_model.eval()
        
        
        # Freeze binding model
        for p in self.binding_model.parameters():
            p.requires_grad = False

        self.immunogenicity_model = ImmunogenicityBetaModel(imm_config, device=device)
        self.immunogenicity_model.train()
    
    def forward(self, peptides, hla_pseudoseqs, hla_prots, *args, target_labels=None):
        binding_score_logit, (joint_embeddings, binding_peptides_embs, binding_hlas_embs) = self.binding_model(peptides, hla_pseudoseqs, hla_prots)
        output, imm_joint_embs = self.immunogenicity_model(binding_score_logit, binding_peptides_embs, binding_hlas_embs, target_labels=target_labels)
        predictions = torch.hstack([torch.sigmoid(binding_score_logit).view(-1,1), output])
        if target_labels is None:
            return predictions
        else:
            return predictions, imm_joint_embs