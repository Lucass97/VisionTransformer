import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    Input Embedding for Vision Transformer (ViT).

    This module splits an image into patches, embeds them into a latent space, 
    and adds positional encoding along with a class token.
    """

    def __init__(self, img_height, img_width, patch_size, n_channels, latent_size) -> None:
        """
        Initializes the Input Embedding module.

        Args:
            img_height (int): Height of the input image.
            img_width (int): Width of the input image.
            patch_size (int): Size of each patch.
            n_channels (int): Number of image channels (e.g., 3 for RGB).
            latent_size (int): Dimensionality of the latent space.
        """
        super(InputEmbedding, self).__init__()

        # Parameters
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.n_channels = n_channels

        self.input_size = patch_size * patch_size * n_channels
        self.num_patches = (img_height // patch_size) * \
            (img_width // patch_size)

        # Learnable class token and positional embeddings
        self.class_token = nn.Parameter(torch.randn(1, 1, latent_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, latent_size))

        # Layers
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(self.input_size, latent_size)

         # Initialize weights
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        """
        Forward pass of the Input Embedding module.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, n_channels, height, width).

        Returns:
            Tensor: Embedded patches with positional encoding and class token.
        """
        batch_size = x.size(0)
        cls_tokens = self.class_token.expand(batch_size, -1, -1)

        # Patchify the image using unfold operator
        patches = self.unfold(x).transpose(1, 2)

        # Compute patch embeddings and add class token
        patch_embeddings = self.linear(patches)
        patch_embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)

        return patch_embeddings + self.pos_embedding
