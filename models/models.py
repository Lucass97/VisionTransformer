from misc.utils.model_utility import calculate_cnn_output_dims
from models import *
from models.resnet.resnet import ResNetBackbone, ResNetClassifier
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

        feature_extractor = None

        if cfg.model.feature_extractor and cfg.model.feature_extractor.type == 'resnet':

            feature_extractor = ResNetBackbone(cfg.data.n_channels,
                                               cfg.model.feature_extractor.num_blocks,
                                               cfg.model.feature_extractor.block_channels)
            
            LOGGER.info(f"Feature extractor instantiated: {feature_extractor.__class__.__name__}.")
            
        n_channels, img_width, img_height = calculate_cnn_output_dims(feature_extractor,
                                                                      cfg.data.n_channels,
                                                                      cfg.data.img_height,
                                                                      cfg.data.img_width)

        input_embedder = InputEmbedding(img_height,
                                        img_width,
                                        cfg.model.patch_size,
                                        n_channels,
                                        cfg.model.latent_size).to(cfg.device)
        
        LOGGER.info(f"Input embedder instantiated : {input_embedder.__class__.__name__}.")
        
        model = ViT(feature_extractor,
                    input_embedder,
                    cfg.model.num_encoders,
                    cfg.model.latent_size,
                    cfg.model.num_heads,
                    cfg.data.num_classes,
                    cfg.model.dropout).to(cfg.device)
    
    elif cfg.model.type == 'resnet':

        model = ResNetClassifier(cfg.data.n_channels,
                                 cfg.model.num_blocks,
                                 cfg.model.block_channels,
                                 cfg.data.num_classes,
                                 cfg.model.dropout).to(cfg.device)
    
    else:
        LOGGER.erro(f"Invalid model type '{cfg.model.type}'. Valid options are 'vit' or 'resnet'.")
        exit()
    
    LOGGER.info(f"Model instantiated: {model.__class__.__name__}")
    
    return model
