import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    Convolutional Neural Network (CNN) Backbone.

    This module consists of a simple convolutional block, including a 
    convolutional layer, ReLU activation, and max pooling. It is designed 
    to process input images and extract feature representations.
    """

    def __init__(self, n_channels: int, out_channels: int = 64) -> None:
        """
        Initializes the CNN backbone.

        Args:
            n_channels (int): Number of input channels (e.g., 3 for RGB images).
            out_channels (int, optional): Number of output feature channels. Default is 64.
        """
        super(CNNBackbone, self).__init__()

        self.n_channels = n_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass of the CNN backbone.

        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width).

        Returns:
            Tensor: Processed feature map after convolution and pooling.
        """
        return self.conv_block(x)
