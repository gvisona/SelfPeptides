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
            self.aa_embs = nn.Embedding.from_pretrained(torch.tensor(aa_embeddings).float()).to(self.device)
        else:
            self.aa_embs = nn.Embedding(len(vocab), config["embedding_dim"], device=device, padding_idx=vocab.index(PADDING_TOKEN))
        self.transformer_encoder = TransformerEncoder(config, device=device)
        
        pma_ln = config.get("PMA_ln", True)
        self.pooling = PMA_MaskedAttention(config["embedding_dim"], config["PMA_num_heads"], 1, ln=pma_ln)
        self.pooling.to(device)
        
    def forward(self, X):
        input_ids, padding_mask = self.tokenizer(X)
        aa_embeddings = self.aa_embs(input_ids)
        token_embeddings = self.transformer_encoder(aa_embeddings, padding_mask)
        return self.pooling(token_embeddings, padding_mask)
    
    
    
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
        self.classifier = nn.Sequential(nn.Linear(config["embedding_dim"], 1), nn.Tanh())
        
    def forward(self, X):
        return self.classifier(self.embedder(X))