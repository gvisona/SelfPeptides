import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.ndimage import gaussian_filter1d, convolve1d
from collections import Counter
from scipy.signal.windows import triang
import numpy as np
from selfpeptide.utils.training_utils import calibrate_mean_var


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

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.01)
            module.weight.data.normal_(mean=1.0, std=0.01)
            
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
        output_dim = config.get("mlp_output_dim", config.get("output_dim", None))
        if input_dim is None:
            input_dim = 2*config["embedding_dim"]
        if input_dim==config["mlp_hidden_dim"]:    
            self.projection_model = ResMLP(config["mlp_num_layers"], config["mlp_hidden_dim"], config["output_dim"], config["dropout_p"])
        else:
            self.projection_model = nn.Sequential(nn.Linear(input_dim, config["mlp_hidden_dim"]),
                                                  ResMLP(config["mlp_num_layers"], config["mlp_hidden_dim"], output_dim))
        self.projection_model.to(device)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.01)
            module.weight.data.normal_(mean=1.0, std=0.01)
        
    def forward(self, *args):
        joined_embs = torch.cat(args, dim=-1)
        if joined_embs.dim()==1:
            joined_embs = joined_embs.unsqueeze(0)
        return self.projection_model(joined_embs)
    
        
        
#######################################
# SET TRANSFORMER MODULES


class MAB_MaskedAttention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB_MaskedAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V, bias=False)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=False)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=False)
        self.ln = ln
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        
        self.mask_val=1.0e10 
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc_k.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc_v.weight, gain=1.0)

        nn.init.kaiming_normal_(self.fc_o.weight, mode='fan_in', nonlinearity='relu')
        
        self.fc_o.bias.data.normal_(mean=0.0, std=0.01)
        
        if self.ln:
            self.ln0.bias.data.normal_(mean=0.0, std=0.01)
            self.ln0.weight.data.normal_(mean=1.0, std=0.01)
            self.ln1.bias.data.normal_(mean=0.0, std=0.01)
            self.ln1.weight.data.normal_(mean=1.0, std=0.01)
        
            
    def forward(self, Q, K, padding_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if padding_mask is not None:
            M = padding_mask.int().repeat(self.num_heads, 1).unsqueeze(1)
            A = torch.softmax((Q_.bmm(K_.transpose(1,2)) - self.mask_val * M)/math.sqrt(self.dim_V), 2)
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



class FDS(nn.Module):

    def __init__(self, fds_feature_dim=512, IR_n_bins=50, fds_start_bin=0,
                 fds_kernel='gaussian', fds_ks=5, fds_sigma=2, fds_momentum=0.9, device="cpu", **kwargs):
        super(FDS, self).__init__()
        self.feature_dim = fds_feature_dim
        self.bucket_num = IR_n_bins
        self.bucket_start = fds_start_bin
        self.device = device
        self.kernel_window = self._get_kernel_window(fds_kernel, fds_ks, fds_sigma)
        self.half_ks = (fds_ks - 1) // 2
        self.momentum = fds_momentum

        self.register_buffer('running_mean', torch.zeros(IR_n_bins - fds_start_bin, fds_feature_dim))
        self.register_buffer('running_var', torch.ones(IR_n_bins - fds_start_bin, fds_feature_dim))
        self.register_buffer('running_mean_last_update', torch.zeros(IR_n_bins - fds_start_bin, fds_feature_dim))
        self.register_buffer('running_var_last_update', torch.ones(IR_n_bins - fds_start_bin, fds_feature_dim))
        self.register_buffer('smoothed_mean_last_update', torch.zeros(IR_n_bins - fds_start_bin, fds_feature_dim))
        self.register_buffer('smoothed_var_last_update', torch.ones(IR_n_bins - fds_start_bin, fds_feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(IR_n_bins - fds_start_bin))

    # @staticmethod
    def _get_kernel_window(self, kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).to(self.device)

    def update_smoothed_stats(self):
        self.running_mean_last_update = self.running_mean
        self.running_var_last_update = self.running_var

        self.smoothed_mean_last_update = F.conv1d(
            input=F.pad(self.running_mean_last_update.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_update = F.conv1d(
            input=F.pad(self.running_var_last_update.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_update.zero_()
        self.running_var_last_update.fill_(1)
        self.smoothed_mean_last_update.zero_()
        self.smoothed_var_last_update.fill_(1)
        self.num_samples_tracked.zero_()


    def update_running_stats(self, features, labels):
        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                curr_feats = features[labels <= label]
            elif label == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            self.running_mean[int(label - self.bucket_start)] = \
                (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
            self.running_var[int(label - self.bucket_start)] = \
                (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]

        # print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels):
        if labels.dim()>1:
            labels = labels.squeeze(1)
        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                features[labels <= label] = calibrate_mean_var(
                    features[labels <= label],
                    self.running_mean_last_update[int(label - self.bucket_start)],
                    self.running_var_last_update[int(label - self.bucket_start)],
                    self.smoothed_mean_last_update[int(label - self.bucket_start)],
                    self.smoothed_var_last_update[int(label - self.bucket_start)])
            elif label == self.bucket_num - 1:
                features[labels >= label] = calibrate_mean_var(
                    features[labels >= label],
                    self.running_mean_last_update[int(label - self.bucket_start)],
                    self.running_var_last_update[int(label - self.bucket_start)],
                    self.smoothed_mean_last_update[int(label - self.bucket_start)],
                    self.smoothed_var_last_update[int(label - self.bucket_start)])
            else:
                features[labels == label] = calibrate_mean_var(
                    features[labels == label],
                    self.running_mean_last_update[int(label - self.bucket_start)],
                    self.running_var_last_update[int(label - self.bucket_start)],
                    self.smoothed_mean_last_update[int(label - self.bucket_start)],
                    self.smoothed_var_last_update[int(label - self.bucket_start)])
        return features