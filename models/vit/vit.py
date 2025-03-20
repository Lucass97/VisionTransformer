import torch.nn as nn

from models.vit.components import EncoderBlock


class ViT(nn.Module):
    """
    Vision Transformer (ViT) implementation.

    This model consists of an embedding layer, multiple encoder blocks, 
    and a classification head. It also includes a mechanism to extract 
    attention weights from the encoder layers.
    """

    def __init__(self, feature_extractor: nn.Module, input_embedder: nn.Module, num_encoders: int,
                 latent_size: int, num_heads: int, num_classes: int, dropout: float) -> None:
        """
        Initializes the Vision Transformer model.

        Args:
            input_embedder: Module responsible for embedding input tokens.
            num_encoders (int): Number of encoder layers.
            latent_size (int): Dimensionality of the latent space.
            num_heads (int): Number of attention heads in each encoder block.
            num_classes (int): Number of output classes for classification.
            dropout (float): Dropout rate for regularization.
        """
        super(ViT, self).__init__()

        self.num_encoders = num_encoders
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.feature_extractor = feature_extractor
        self.input_embedder = input_embedder

        self.encoder_stack = nn.ModuleList([
            EncoderBlock(latent_size, num_heads, dropout)
            for _ in range(num_encoders)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, latent_size),
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(latent_size, num_classes),
        )

        self.attention_weights = {}
        self.register_attention_hooks()

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Classification logits.
        """
        if self.feature_extractor:
            x = self.feature_extractor(x)
            
        enc_output = self.input_embedder(x)

        for enc_layer in self.encoder_stack:
            enc_output = enc_layer(enc_output)

        cls_token_embed = enc_output[:, 0]

        return self.head(cls_token_embed)

    def register_attention_hooks(self) -> None:
        """
        Registers forward hooks to extract attention weights from each encoder layer.
        """
        for i, encoder in enumerate(self.encoder_stack):
            def hook_fn(module, input, output, layer_idx=i) -> None:
                self.attention_weights[layer_idx] = module.attention_weights
            encoder.register_forward_hook(hook_fn)

    def get_attention_weights(self) -> dict:
        """
        Returns extracted attention weights.

        Returns:
            dict: A dictionary mapping layer indices to attention weights.
        """
        return self.attention_weights
