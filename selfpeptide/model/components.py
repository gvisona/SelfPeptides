import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.ndimage import gaussian_filter1d, convolve1d
from collections import Counter
from scipy.signal.windows import triang
import numpy as np
from selfpeptide.utils.training_utils import calibrate_mean_var
import torch.nn as nn
import torch.nn.functional as F


class ResNorm(nn.Module):
    """
    Residual Normalization Module
    """

    def __init__(self, embedding_dim: int):
        """
        Initialize the ResNorm module

        Args:
            embedding_dim (int): The dimension of the input embedding
        """
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, res_connection_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNorm module

        Args:
            x (torch.Tensor): The input tensor
            res_connection_x (torch.Tensor): The residual connection tensor

        Returns:
            torch.Tensor: The output tensor after applying layer normalization to the sum of x and res_connection_x
        """
        add = x + res_connection_x
        return self.norm(add)


class ResBlock(nn.Module):
    """
    Residual Block Module
    """

    def __init__(self, dim: int, p_dropout: float = 0.2):
        """
        Initialize the ResBlock module

        Args:
            dim (int): The dimension of the input and output tensors
            p_dropout (float, optional): The dropout probability. Defaults to 0.2.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(p_dropout)
        )
        self.layer_norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights of the linear and layer normalization layers

        Args:
            module (nn.Module): The module to initialize the weights for
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.01)
            module.weight.data.normal_(mean=1.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlock module

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor after applying layer normalization and adding the residual connection
        """
        return self.layer_norm(x + self.block(x))


class ResMLP(nn.Module):
    """
    Residual Multi-Layer Perceptron Module
    """

    def __init__(self, num_layers: int, dim: int, output_dim: int, p_dropout: float = 0.2):
        """
        Initialize the ResMLP module

        Args:
            num_layers (int): The number of layers in the MLP
            dim (int): The dimension of the input and output tensors
            output_dim (int): The dimension of the output tensor
            p_dropout (float, optional): The dropout probability. Defaults to 0.2.
        """
        super().__init__()
        layers = [ResBlock(dim, p_dropout=p_dropout) for _ in range(num_layers)] + [
            nn.Linear(dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResMLP module

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor after passing through the MLP
        """
        return self.net(x)


class ResMLP_Network(nn.Module):
    """
    Residual MLP Network Module
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize the ResMLP_Network module

        Args:
            config (dict): The configuration dictionary
            device (str, optional): The device to run the module on. Defaults to "cpu".
        """
        super().__init__()
        self.config = config
        self.device = device

        input_dim = config.get("mlp_input_dim", None)
        output_dim = config.get("mlp_output_dim", config.get("output_dim", None))
        if input_dim is None:
            input_dim = 2 * config["embedding_dim"]
        if input_dim == config["mlp_hidden_dim"]:
            self.projection_model = ResMLP(config["mlp_num_layers"], config["mlp_hidden_dim"], config["output_dim"],
                                           config["dropout_p"])
        else:
            self.projection_model = nn.Sequential(nn.Linear(input_dim, config["mlp_hidden_dim"]),
                                                  ResMLP(config["mlp_num_layers"], config["mlp_hidden_dim"],
                                                          output_dim))
        self.projection_model.to(device)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights of the linear and layer normalization layers

        Args:
            module (nn.Module): The module to initialize the weights for
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.01)
            module.weight.data.normal_(mean=1.0, std=0.01)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResMLP_Network module

        Args:
            *args (torch.Tensor): The input tensors

        Returns:
            torch.Tensor: The output tensor after passing through the projection model
        """
        joined_embs = torch.cat(args, dim=-1)
        if joined_embs.dim() == 1:
            joined_embs = joined_embs.unsqueeze(0)
        return self.projection_model(joined_embs)


class MAB_MaskedAttention(nn.Module):
    """
    Masked Attention Block Module
    """

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False):
        """
        Initialize the MAB_MaskedAttention module

        Args:
            dim_Q (int): The dimension of the query tensor
            dim_K (int): The dimension of the key tensor
            dim_V (int): The dimension of the value tensor
            num_heads (int): The number of attention heads
            ln (bool, optional): Whether to apply layer normalization. Defaults to False.
        """
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

        self.mask_val = 1.0e10

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize the weights of the linear layers

        Args:
            module (nn.Module): The module to initialize the weights for
        """
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

    def forward(self, Q: torch.Tensor, K: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the MAB_MaskedAttention module

        Args:
            Q (torch.Tensor): The query tensor
            K (torch.Tensor): The key tensor
            padding_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying masked attention
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if padding_mask is not None:
            M = padding_mask.int().repeat(self.num_heads, 1).unsqueeze(1)
            A = torch.softmax((Q_.bmm(K_.transpose(1, 2)) - self.mask_val * M) / math.sqrt(self.dim_V), 2)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class PMA_MaskedAttention(nn.Module):
    """
    Projected Masked Attention Module
    """

    def __init__(self, dim: int, num_heads: int, num_seeds: int, ln: bool = False):
        """
        Initialize the PMA_MaskedAttention module

        Args:
            dim (int): The dimension of the input and output tensors
            num_heads (int): The number of attention heads
            num_seeds (int): The number of seed vectors
            ln (bool, optional): Whether to apply layer normalization. Defaults to False.
        """
        super(PMA_MaskedAttention, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB_MaskedAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the PMA_MaskedAttention module

        Args:
            X (torch.Tensor): The input tensor
            padding_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying masked attention
        """
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, padding_mask=padding_mask).squeeze(1)


class FDS(nn.Module):
    """
    Feature Distribution Smoothing Module
    """

    def __init__(self, fds_feature_dim: int = 512, IR_n_bins: int = 50, fds_start_bin: int = 0,
                 fds_kernel: str = 'gaussian', fds_ks: int = 5, fds_sigma: int = 2, fds_momentum: float = 0.9,
                 device: str = "cpu", **kwargs):
        """
        Initialize the FDS module

        Args:
            fds_feature_dim (int, optional): The dimension of the input features. Defaults to 512.
            IR_n_bins (int, optional): The number of bins in the intensity range. Defaults to 50.
            fds_start_bin (int, optional): The starting bin index. Defaults to 0.
            fds_kernel (str, optional): The kernel type for smoothing. Defaults to 'gaussian'.
            fds_ks (int, optional): The kernel size. Defaults to 5.
            fds_sigma (int, optional): The sigma value for the kernel. Defaults to 2.
            fds_momentum (float, optional): The momentum value for updating running statistics. Defaults to 0.9.
            device (str, optional): The device to run the module on. Defaults to "cpu".
        """
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

    def _get_kernel_window(self, kernel: str, ks: int, sigma: int) -> torch.Tensor:
        """
        Get the kernel window for smoothing

        Args:
            kernel (str): The kernel type
            ks (int): The kernel size
            sigma (int): The sigma value for the kernel

        Returns:
            torch.Tensor: The kernel window tensor
        """
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(
                gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(
                map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).to(self.device)

    def update_smoothed_stats(self) -> None:
        """
        Update the smoothed statistics
        """
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

    def reset(self) -> None:
        """
        Reset the running statistics
        """
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_update.zero_()
        self.running_var_last_update.fill_(1)
        self.smoothed_mean_last_update.zero_()
        self.smoothed_var_last_update.fill_(1)
        self.num_samples_tracked.zero_()

    def update_running_stats(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the running statistics

        Args:
            features (torch.Tensor): The input features tensor
            labels (torch.Tensor): The labels tensor
        """
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
