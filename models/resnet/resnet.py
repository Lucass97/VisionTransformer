import torch.nn as nn
import torch.nn.functional as F

from models.resnet.components import ResidualBlock


class ResNetBackbone(nn.Module):
    
    def __init__(self, input_channels: int, num_blocks: list, block_channels: int) -> None:
        """
        Initialize the ResNet model.

        Args:
            input_channels (int): Number of input channels in the image (e.g., 3 for RGB).
            num_blocks (list): List containing the number of residual blocks in each stage.
            block_channels (int): The number of channels in the first block.
        """
        super(ResNetBackbone, self).__init__()

        self.block_channels = block_channels  # Initial number of channels
        self.num_blocks = num_blocks

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

        self.last_out_channels = in_channels

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
        
        return x


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier model.

    This model utilizes a ResNet backbone for feature extraction followed by 
    a classification head with global average pooling and a fully connected layer.
    """
    
    def __init__(self, input_channels: int, num_blocks: list, block_channels: int, num_classes: int, dropout: float = 0.0):
        """
        Initialize the ResNetClassifier model.

        Args:
            input_channels (int): Number of input channels in the image (e.g., 3 for RGB images).
            num_blocks (list): List specifying the number of residual blocks in each stage of the ResNet backbone.
            block_channels (int): Number of channels in the first block; it increases in deeper layers.
            num_classes (int): Number of target classes for classification.
            dropout (float): Dropout rate applied before the final classification layer.
        """
        super(ResNetClassifier, self).__init__()

        # Initialize the ResNet backbone for feature extraction
        self.resnet_backbone = ResNetBackbone(
            input_channels=input_channels,
            num_blocks=num_blocks,
            block_channels=block_channels
        )
        
        self.last_out_channels = self.resnet_backbone.last_out_channels

        # Global average pooling layer to reduce spatial dimensions to a single value per channel
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.last_out_channels, num_classes)
        )

    def forward(self, x):
        """
        Perform a forward pass through the ResNet classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor containing class logits of shape (batch_size, num_classes).
        """
        x = self.resnet_backbone(x) 
        x = self.global_pool(x)
        return self.head(x)
    