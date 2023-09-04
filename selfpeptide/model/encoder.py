import torch
import torch.nn as nn
import math
import re
from selfpeptide.model.components import *


class AA_Tokenizer(nn.Module):
    def __init__(self, sorted_vocabulary, device="cpu", padding_token="*"):
        super().__init__()
        self.device = device
        self.padding_token = padding_token
        
            
        self.sorted_vocabulary = sorted_vocabulary.copy()
        if padding_token not in sorted_vocabulary:
            self.sorted_vocabulary.append(padding_token)
            
        self.idx2token = {i: a for i, a in enumerate(self.sorted_vocabulary)}
        self.token2idx = {a: i for i, a in self.idx2token.items()}
        
    def _pad_sequences(self, seqs):
        maxlen = max([len(s) for s in seqs])
        padded_seqs = []
        for s in seqs:
            npad = maxlen - len(s)
            padded_seqs.append(s + ''.join([self.padding_token]*npad))
        return padded_seqs
        
    
    def forward(self, seqs):
        if isinstance(seqs, (list, tuple)):
            seqs = ["".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs]    
            seqs = self._pad_sequences(seqs)
            attention_mask = []
            aa_ids = []
            for s in seqs:
                attention_mask.append([1 if aa==self.padding_token else 0 for aa in s]) # Mask is 1 for padding token, 0 otherwise
                aa_ids.append([self.token2idx[aa] for aa in s])
            attention_mask = torch.tensor(attention_mask).bool().to(self.device) 
            aa_ids = torch.LongTensor(aa_ids).to(self.device) 
        elif torch.is_tensor(seqs):
            aa_ids = seqs.long().to(self.device)
            attention_mask = (torch.eq(aa_ids, self.token2idx[self.padding_token])).bool().to(self.device) 
        else:
            raise ValueError("AA_Tokenizer requires strings of amino acids or pre-tokenized tensors")
        
        return aa_ids, attention_mask
    
    
    


class TEncoderLayer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        self.multihead_attention = nn.MultiheadAttention(config["embedding_dim"], config["num_heads"], batch_first=True)
        
        self.dropout1 = nn.Dropout(config["dropout_p"])
        self.res_norm1 = ResNorm(config["embedding_dim"])
        self.feed_forward = nn.Sequential(nn.Linear(config["embedding_dim"], config["transf_hidden_dim"]),
                                          nn.ReLU(),
                                          nn.Linear(config["transf_hidden_dim"], config["embedding_dim"]))
        self.dropout2 = nn.Dropout(config["dropout_p"])
        self.res_norm2 = ResNorm(config["embedding_dim"])
        
        self.feed_forward.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.01)
            module.weight.data.normal_(mean=1.0, std=0.01)
    
    def forward(self, X, padding_mask):
        multihead_output, attn_weights = self.multihead_attention(X, X, X, padding_mask)
        multihead_output = self.dropout1(multihead_output)

        resnorm_output = self.res_norm1(multihead_output, X)

        feedforward_output = self.feed_forward(resnorm_output)
        feedforward_output = self.dropout2(feedforward_output)

        return self.res_norm2(resnorm_output, feedforward_output)
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model, device=device, requires_grad=False)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # self.pe = pe

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)        
        

class TransformerEncoder(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.pos_encoding = PositionalEncoding(config["embedding_dim"], config["dropout_p"], max_len=100, device=device)
        self.dropout = nn.Dropout(config["dropout_p"])
        self.encoder_layers = nn.ModuleList([TEncoderLayer(config) for _ in range(config["n_attention_layers"])])
        self.device = device
        
    def forward(self, X, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.zeros_like(X).int().to(self.device)
            
        X = self.pos_encoding(X)
        X = self.dropout(X)

        for i, layer in enumerate(self.encoder_layers):
            X = layer(X, padding_mask)

        return X