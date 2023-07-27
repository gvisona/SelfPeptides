import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNorm(nn.Module):
    # Post LN norm
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, res_connection_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + res_connection_x
    
        # Apply layer normalization to the sum
        return self.norm(add)
    
       
class ResBlock(nn.Module):
    def __init__(self, dim, p_dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(dim, dim), 
            nn.Dropout(p_dropout)
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x + self.block(x))


class ResMLP(nn.Module):
    def __init__(self, num_layers, dim, output_dim, p_dropout=0.2):
        super().__init__()
        layers = [ResBlock(dim, p_dropout=p_dropout) for _ in range(num_layers)] + [
            nn.Linear(dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    

class ResMLP_Network(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.config = config
        self.device = device
        
        input_dim = config.get("mlp_input_dim", None)
        if input_dim is None:
            input_dim = 2*config["embedding_dim"]
        if input_dim==config["mlp_hidden_dim"]:    
            self.projection_model = ResMLP(config["mlp_num_layers"], config["mlp_hidden_dim"], config["output_dim"], config["dropout_p"])
        else:
            self.projection_model = nn.Sequential(nn.Linear(input_dim, config["mlp_hidden_dim"]),
                                                  ResMLP(config["mlp_num_layers"], config["mlp_hidden_dim"], config["output_dim"]))
        self.projection_model.to(device)
        
    def forward(self, *args):
        joined_embs = torch.cat(args, dim=-1)
        if joined_embs.dim()==1:
            joined_embs = joined_embs.unsqueeze(0)
        return self.projection_model(joined_embs)
    
        
        
#######################################
# SET TRANSFORMER MODULES


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O



class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class MAB_MaskedAttention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB_MaskedAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        
        self.mask_val=1.0e10 

    def forward(self, Q, K, padding_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if padding_mask is not None:
            M = padding_mask.repeat(self.num_heads, 1).unsqueeze(1)
            A = torch.softmax((Q_.bmm(K_.transpose(1,2)) - self.mask_val * (1-M))/math.sqrt(self.dim_V), 2)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
    

class PMA_MaskedAttention(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA_MaskedAttention, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB_MaskedAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, padding_mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, padding_mask=padding_mask).squeeze(1)


#
####################################################
