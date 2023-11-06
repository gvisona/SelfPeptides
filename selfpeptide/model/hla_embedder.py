import json
import torch
import torch.nn as nn
import numpy as np

class Pretrained_HLA_Embedder(nn.Module):
    def __init__(self, config, device="cpu", prefix=""):
        super().__init__()
        with open(config[prefix+"seq2idx_file"], "r") as f:
            self.seq2idx = json.load(f)
        self.idx2seq = {i:s for s, i in self.seq2idx.items()}
        self.device = device
        embeddings = np.load(config[prefix+"embeddings_file"])
        self.pretrained_embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings, device=self.device))
        self.pretrained_embeddings.to(device)
    
    def forward(self, hla_seqs):
        idxs = [self.seq2idx[s] for s in hla_seqs]
        idxs_tensor = torch.LongTensor(idxs, device=self.device)
        embs = self.pretrained_embeddings(idxs_tensor)
        return embs
    
    
class Joint_HLA_Embedder(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.pseudoseq_embedder = Pretrained_HLA_Embedder(config, device=device, prefix="pseudoseq_")
        self.protein_embedder = Pretrained_HLA_Embedder(config, device=device, prefix="protein_")
        
        self.projection_layer = nn.Linear(2*config["hla_embedding_dim"], config["embedding_dim"], device=device)
        
    def forward(self, hla_pseudoseqs, hla_protein_seqs):
        pseq_embs = self.pseudoseq_embedder(hla_pseudoseqs)
        prot_embs = self.protein_embedder(hla_protein_seqs)
        joint_embs = torch.cat([pseq_embs, prot_embs], dim=1)
        return self.projection_layer(joint_embs)