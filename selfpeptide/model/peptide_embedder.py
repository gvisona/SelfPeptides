import torch
import numpy as np
from selfpeptide.utils.processing_utils import get_vocabulary_tokens
from selfpeptide.utils.constants import *
from selfpeptide.model.components import PMA_MaskedAttention
from selfpeptide.model.encoder import AA_Tokenizer, TransformerEncoder

import torch.nn as nn

class PeptideEmbedder(nn.Module):
    """
    Class for embedding peptide sequences.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize the PeptideEmbedder.

        Args:
            config (dict): Configuration parameters.
            device (str): Device to run the model on (default: "cpu").
        """
        super().__init__()

        vocab = get_vocabulary_tokens()
        self.vocab = vocab
        self.device = device

        self.tokenizer = AA_Tokenizer(vocab, device=device)

        aa_embeddings = None
        if config.get("pretrained_aa_embeddings", None) is not None and config["pretrained_aa_embeddings"] != "none":
            aa_embeddings = np.load(config["pretrained_aa_embeddings"])

        if aa_embeddings is not None:
            self.aa_embs = nn.Embedding.from_pretrained(torch.tensor(aa_embeddings, requires_grad=False).float(),
                                                        freeze=True).to(self.device)
        else:
            self.aa_embs = nn.Embedding(len(vocab), config["embedding_dim"], device=device,
                                        padding_idx=vocab.index(PADDING_TOKEN))
        self.transformer_encoder = TransformerEncoder(config, device=device)

        pma_ln = config.get("PMA_ln", True)
        self.pooling = PMA_MaskedAttention(config["embedding_dim"], config["PMA_num_heads"], 1, ln=pma_ln)
        self.pooling.to(device)

        self.fn = nn.Linear(config["embedding_dim"], config["embedding_dim"])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PeptideEmbedder.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embedding_dim).
        """
        input_ids, padding_mask = self.tokenizer(X)
        aa_embeddings = self.aa_embs(input_ids)
        token_embeddings = self.transformer_encoder(aa_embeddings, padding_mask)
        sequence_embeddings = self.pooling(token_embeddings, padding_mask)
        return self.fn(sequence_embeddings)


class SelfPeptideEmbedder_withLogits(nn.Module):
    """
    Class for embedding peptide sequences with logits.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize the SelfPeptideEmbedder_withLogits.

        Args:
            config (dict): Configuration parameters.
            device (str): Device to run the model on (default: "cpu").
        """
        super().__init__()

        self.device = device
        self.embedder = PeptideEmbedder(config, device)
        self.classifier = nn.Linear(config["embedding_dim"], 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SelfPeptideEmbedder_withLogits.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.classifier(self.embedder(X))


class SelfPeptideEmbedder_Hinge(nn.Module):
    """
    Class for embedding peptide sequences with hinge loss.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize the SelfPeptideEmbedder_Hinge.

        Args:
            config (dict): Configuration parameters.
            device (str): Device to run the model on (default: "cpu").
        """
        super().__init__()

        self.device = device
        self.embedder = PeptideEmbedder(config, device)
        self.classifier = nn.Sequential(nn.Linear(config["embedding_dim"], config["classifier_hidden_dim"]),
                                        nn.ReLU(),
                                        nn.Linear(config["classifier_hidden_dim"], 1),
                                        nn.Tanh())

        self.classifier.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights of the linear layers.

        Args:
            module (nn.Module): Linear layer module.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.01)
            module.weight.data.normal_(mean=1.0, std=0.01)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SelfPeptideEmbedder_Hinge.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        embeddings = self.embedder(X)
        predictions = self.classifier(embeddings)
        return predictions, embeddings


class SelfPeptideEmbedder_withProjHead(nn.Module):
    """
    Class for embedding peptide sequences with projection head.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize the SelfPeptideEmbedder_withProjHead.

        Args:
            config (dict): Configuration parameters.
            device (str): Device to run the model on (default: "cpu").
        """
        super().__init__()

        self.device = device
        self.embedder = PeptideEmbedder(config, device)
        self.projection_head = nn.Sequential(nn.Linear(config["embedding_dim"], config["projection_hidden_dim"]),
                                             nn.ReLU(),
                                             nn.Linear(config["projection_hidden_dim"], config["projection_dim"]))

        self.register_buffer("human_peptides_cosine_centroid", torch.zeros(config["projection_dim"]).float())

    def forward(self, X: torch.Tensor, return_sns_score: bool = False, eps: float = 1e-8) -> torch.Tensor:
        """
        Forward pass of the SelfPeptideEmbedder_withProjHead.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            return_sns_score (bool): Whether to return the sns_score (default: False).
            eps (float): Small value for numerical stability (default: 1e-8).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, projection_dim) if return_sns_score is False,
                          otherwise a tuple of tensors (projections, embeddings, sns_score).
        """
        embeddings = self.embedder(X)
        projections = self.projection_head(embeddings)
        if return_sns_score:
            projections_n = projections.norm(dim=1)[:, None]
            projections_norm = projections / torch.clamp(projections_n, min=eps)
            sns_score = torch.mm(projections_norm, self.human_peptides_cosine_centroid.view(-1, 1))
            return projections, embeddings, sns_score
        return projections, embeddings