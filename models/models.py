from models import *
from models.resnet.resnet import ResNetClassifier
from models.vit.embedder import InputEmbedding
from models.vit.vit import ViT


def build_model(cfg):
    '''
    Instantiates a model based on the configuration (`cfg.model.type`).
    Logs the model type, input embedder, and class name for the instantiated model.
    
    Args:
        cfg (Namespace): Configuration containing model settings.
    
    Returns:
        object: The instantiated model (ViT or ResNet).
    '''
    
    model = None

    if cfg.model.type == 'vit':

        input_embedder = InputEmbedding(cfg.data.img_height,
                                        cfg.data.img_width,
                                        cfg.model.patch_size,
                                        cfg.data.n_channels,
                                        cfg.model.latent_size).to(cfg.device)
        
        LOGGER.info(f"Instantiating a Vision Transformer model with InputEmbedding: \
                    {input_embedder.__class__.__name__}.")
        
        model = ViT(input_embedder,
                    cfg.model.num_encoders,
                    cfg.model.latent_size,
                    cfg.model.num_heads,
                    cfg.data.num_classes,
                    cfg.model.dropout).to(cfg.device)
        
        LOGGER.info(f"Model instantiated: {model.__class__.__name__}")
    
    elif cfg.model.type == 'resnet':

        model = ResNetClassifier(cfg.data.n_channels,
                                 cfg.model.num_blocks,
                                 cfg.model.block_channels,
                                 cfg.data.num_classes,
                                 cfg.model.dropout).to(cfg.device)
        
        LOGGER.info(f"Model instantiated: {model.__class__.__name__}")
    
    else:
        LOGGER.error(f"Invalid model type '{cfg.model.type}'. Valid options are 'vit' or 'resnet'.")
        return None
    
    return model
