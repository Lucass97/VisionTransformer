import os
from typing import List, Literal, Union
import yaml
from pydantic import BaseModel, Field

from misc.utils.generic_utility import generate_experiment_name


class DataConfig(BaseModel):
    name: str
    class_path: str
    base_path: str
    img_height: int = Field(..., gt=0)
    img_width: int = Field(..., gt=0)
    n_channels: int = Field(..., ge=1)
    num_classes: int = Field(..., ge=2)


class ViTConfig(BaseModel):
    type: Literal["vit"] = "vit"
    patch_size: int = Field(..., gt=0)
    latent_size: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    num_encoders: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, le=1.0)


class ResNetConfig(BaseModel):
    type: Literal["resnet"] = "resnet" 
    num_blocks: List[int]
    block_channels: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    epochs: int = Field(..., gt=0)
    lr: float = Field(..., gt=0.0)
    weight_decay: float
    batch_size: int = Field(..., gt=0)


class ModelCheckpoint(BaseModel):
    base_path: str
    save_freq: int


class LoggerConfig(BaseModel):
    base_path: str
    step: int = Field(..., gt=0)
    max_grid_dim: int = Field(..., gt=0)


class Config(BaseModel):
    device: str
    data: DataConfig
    model: Union[ViTConfig, ResNetConfig] = Field(..., discriminator="type")
    model_checkpoint: ModelCheckpoint
    training: TrainingConfig
    logger: LoggerConfig


def load_config(yaml_path: str, experiment_name: str) -> tuple[Config, str]:
    """
    Loads a YAML configuration file and validates it using Pydantic.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        experiment_name (str): Name of the experiment. If not provided, it is generated automatically.

    Returns:
        tuple[Config, str]: The validated configuration object and the experiment name.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"The configuration file '{yaml_path}' does not exist!")

    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    cfg = Config(**params)

    # If the experiment name is not provided, generate one based on the dataset name and timestamp
    experiment_name = experiment_name if experiment_name else generate_experiment_name(
        cfg.data.name)
    
    cfg.model_checkpoint.base_path = os.path.join(cfg.model_checkpoint.base_path, experiment_name)
    cfg.logger.base_path = os.path.join(cfg.logger.base_path, experiment_name)

    return cfg, experiment_name
