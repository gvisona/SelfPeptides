import torch
import torch.nn as nn
import numpy as np
from selfpeptide.utils.processing_utils import get_vocabulary_tokens
from selfpeptide.utils.constants import *
from selfpeptide.model.components import PMA_MaskedAttention
from selfpeptide.model.encoder import AA_Tokenizer, TransformerEncoder
        
        

class PeptideEmbedder(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        
        vocab = get_vocabulary_tokens()
        self.vocab = vocab
        self.device = device
        
        self.tokenizer = AA_Tokenizer(vocab, device=device)
        
        aa_embeddings = None
        if config.get("pretrained_aa_embeddings", None) is not None and config["pretrained_aa_embeddings"]!="none":
            aa_embeddings = np.load(config["pretrained_aa_embeddings"])

        if aa_embeddings is not None:
            self.aa_embs = nn.Embedding.from_pretrained(torch.tensor(aa_embeddings, requires_grad=False).float(), freeze=True).to(self.device)
        else:
            self.aa_embs = nn.Embedding(len(vocab), config["embedding_dim"], device=device, padding_idx=vocab.index(PADDING_TOKEN))
        self.transformer_encoder = TransformerEncoder(config, device=device)
        
        pma_ln = config.get("PMA_ln", True)
        self.pooling = PMA_MaskedAttention(config["embedding_dim"], config["PMA_num_heads"], 1, ln=pma_ln)
        self.pooling.to(device)
        
        self.fn = nn.Linear(config["embedding_dim"], config["embedding_dim"])


    def forward(self, X):
        input_ids, padding_mask = self.tokenizer(X)
        aa_embeddings = self.aa_embs(input_ids)
        token_embeddings = self.transformer_encoder(aa_embeddings, padding_mask)
        sequence_embeddings = self.pooling(token_embeddings, padding_mask)
        return self.fn(sequence_embeddings)
    
    
    
class SelfPeptideEmbedder_withLogits(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        
        self.device = device
        self.embedder = PeptideEmbedder(config, device)
        self.classifier = nn.Linear(config["embedding_dim"], 1)
        
    def forward(self, X):
        return self.classifier(self.embedder(X))
    

    
class SelfPeptideEmbedder_Hinge(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        
        self.device = device
        self.embedder = PeptideEmbedder(config, device)
        self.classifier = nn.Sequential(nn.Linear(config["embedding_dim"], config["classifier_hidden_dim"]),
                                        nn.ReLU(),
            nn.Linear(config["classifier_hidden_dim"], 1), 
            nn.Tanh())
        
        self.classifier.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.01)
            module.weight.data.normal_(mean=1.0, std=0.01)
            
    def forward(self, X):
        embeddings = self.embedder(X)
        predictions = self.classifier(embeddings)
        return predictions, embeddings
    
    

class SelfPeptideEmbedder_withProjHead(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        
        self.device = device
        self.embedder = PeptideEmbedder(config, device)
        self.projection_head = nn.Sequential(nn.Linear(config["embedding_dim"], config["projection_hidden_dim"]),
                                        nn.ReLU(),
            nn.Linear(config["projection_hidden_dim"], config["projection_dim"]))
        
        self.register_buffer("human_peptides_cosine_centroid", torch.zeros(config["projection_dim"]).float())


        
    def forward(self, X, return_sns_score=False, eps=1e-8):
        embeddings = self.embedder(X)
        projections = self.projection_head(embeddings)
        if return_sns_score:
            embeddings_n = embeddings.norm(dim=1)[:, None]
            embeddings_norm = embeddings / torch.clamp(embeddings_n, min=eps)
            sns_score = torch.mm(embeddings_norm, self.human_peptides_cosine_centroid.view(-1,1))
            return projections, embeddings, sns_score
        return projections, embeddings
    
    