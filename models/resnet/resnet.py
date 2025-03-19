import torch.nn as nn
import torch.nn.functional as F

from models.resnet.components import ResidualBlock


class ResNet(nn.Module):
    
    def __init__(self, input_channels: int, num_blocks: list, block_channels: int, num_classes: int, dropout: float):
        """
        Initialize the ResNet model.

        Args:
            input_channels (int): Number of input channels in the image (e.g., 3 for RGB).
            num_blocks (list): List containing the number of residual blocks in each stage.
            block_channels (int): The number of channels in the first block.
            num_classes (int): Number of output classes for classification.
            dropout (float): Dropout rate to apply before the final fully connected layer.
        """
        super(ResNet, self).__init__()

        self.block_channels = block_channels  # Initial number of channels
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.dropout = dropout

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, self.block_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.block_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.residual_stack = nn.ModuleList()
        
        in_channels = self.block_channels

        # Building the residual blocks for each stage
        for stage, num_block in enumerate(self.num_blocks):
            
            # Double the number of channels after the first stage
            out_channels = in_channels * 2 if stage > 0 else in_channels
            
            for i in range(num_block):
                self.residual_stack.append(
                    ResidualBlock(
                        in_channels=in_channels if i == 0 else out_channels, 
                        out_channels=out_channels,
                        downsample=(i == 0 and in_channels != out_channels)  # Downsample if required
                    )
                )

            # Update the in_channels for the next stage
            in_channels = out_channels

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)  # Fully connected layer for classification
        )

    def forward(self, x):
        """
        Forward pass through the entire ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (logits for classification).
        """

        x = self.initial_conv(x)
        
        # Pass through residual blocks
        for res_block in self.residual_stack:
            x = res_block(x)
        
        x = self.global_pool(x)
        
        return self.head(x)
