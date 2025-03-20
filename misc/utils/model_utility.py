from misc.utils.img_utility import apply_color_palette
import torch
import torch.nn.functional as F


def calculate_cnn_output_dims(cnn_backbone, input_channels: int, img_height: int, img_width: int) -> tuple[int, int, int]:
    """
    Calculates the output dimensions (height, width, channels) of the CNN model
    given the input image's height and width.

    Args:
        cnn_backbone: The CNN model (e.g., a pre-trained or custom network).
        input_channels (int): The number of channels in the input image (e.g., 3 for RGB images).
        img_height (int): The height of the input image.
        img_width (int): The width of the input image.

    Returns:
        tuple[int, int, int]: A tuple representing the output dimensions (output_height, output_width, output_channels).
        - output_height: The height of the output image after processing through the CNN model.
        - output_width: The width of the output image after processing through the CNN model.
        - output_channels: The number of channels in the output image (depends on the CNN model's output channels).
    """

    if cnn_backbone is None:
        return input_channels, img_height, img_width

    device = next(cnn_backbone.parameters()).device
    
    with torch.no_grad():
        dummy_input = torch.zeros(1, input_channels, img_height, img_width, device=device)
        output = cnn_backbone(dummy_input)
    
    return output.shape[1], output.shape[2], output.shape[3]


def reconstruct_attn_from_patches(attn_maps, img_size, patch_size):
    """
    This function reconstructs an attention map into an image-compatible tensor.
    
    Parameters:
    attn_maps (Tensor): The attention tensor with shape (batch, num_tokens, num_tokens).
    img_size (tuple): A tuple (height, width) representing the original image dimensions.
    patch_size (int): The size of each patch used during tokenization.
    
    Returns:
    Tensor: The processed attention map resized to match the original image dimensions.
    """
    
    last_layer = list(attn_maps.keys())[-1]
    attn_map = attn_maps[last_layer]
    attn_map = attn_map.clone().detach()
    batch_size, num_tokens, _ = attn_map.shape
    grid_height, grid_width = img_size[0] // patch_size, img_size[1] // patch_size
    
    # Exclude the CLS token (first row and column)
    attn_map = attn_map[:, 1:, 1:]
    
    # Compute the mean over the patch dimension
    attn_map = attn_map.mean(dim=1)
    
    attn_map = attn_map.view(batch_size, 1, grid_height, grid_width)
    
    attn_map = F.interpolate(attn_map, size=img_size, mode='bilinear', align_corners=False)
    
    attn_map = apply_color_palette(attn_map)
    
    return attn_map