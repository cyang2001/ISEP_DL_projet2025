import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Implémente l'encodage positionnel classique de Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # dimension paire
        pe[:, 1::2] = torch.cos(position * div_term)  # dimension impaire

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return x


class SignLanguageTransformer(nn.Module):
    """
    Transformer pour la reconnaissance de signes à partir des séquences vidéo.
    """
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)  # [B, T, D_in] → [B, T, D_model]
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)  # [B, D_model] → [B, num_classes]

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, D_in]
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        x = self.input_proj(x)               # [B, T, D_model]
        x = self.pos_encoder(x)              # add positional encoding
        x = self.transformer_encoder(x)      # [B, T, D_model]
        x = x.mean(dim=1)                    # average pooling over T → [B, D_model]
        logits = self.classifier(x)          # → [B, num_classes]
        return logits