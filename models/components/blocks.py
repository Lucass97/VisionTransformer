import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    This module consists of a multi-head self-attention mechanism followed by 
    a feed-forward neural network with residual connections and layer normalization.
    """

    def __init__(self, latent_size, num_heads, dropout) -> None:
        """
        Initializes the Encoder Block.

        Args:
            latent_size (int): Dimensionality of the latent space.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for regularization.
        """
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(self.latent_size)
        self.norm2 = nn.LayerNorm(self.latent_size)

        self.multihead = nn.MultiheadAttention(
            latent_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.enc_MLP = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 4, latent_size),
        )

        self.attention_weights = None

    def forward(self, x):
        """
        Forward pass of the Encoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, latent_size).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Multi-Head Self-Attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.multihead(x_norm, x_norm, x_norm)
        self.attention_weights = attn_weights
        x = x + attn_output

        # Feed-Forward Network
        x_norm = self.norm2(x)
        x = x + self.enc_MLP(x_norm)

        return x
