import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from misc.configs import Config
from misc.logger.logger import CustomLogger
from misc.logger.tensorboard import TensorboardLogger
from misc.utils.generic_utility import processing
from misc.utils.model_utility import calculate_cnn_output_dims, reconstruct_attn_from_patches

from dataset.datasets import create_dataset_instance

from models.components.components import CNNBackbone
from models.components.embedder import InputEmbedding
from models.models import ViT


class Trainer:

    def __init__(self, cfg: Config, weights_path: str) -> None:
        """
        Initializes the Trainer class with model configuration and experiment settings.

        Args:
            cfg (Config): A configuration object containing hyperparameters and settings.
        """

        self.cfg = cfg
        self.weights_path = weights_path

        # Dataset creation
        self.train_set = create_dataset_instance(cfg.data.class_path, cfg.data.base_path, train=True)
        self.test_set = create_dataset_instance(cfg.data.class_path, cfg.data.base_path, train=False)

        # DataLoader creation
        self.train_loader = DataLoader(self.train_set, cfg.training.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_set, cfg.training.batch_size, shuffle=True, num_workers=4)

        # Model components initialization
        self.cnn_backbone = CNNBackbone(n_channels=cfg.data.n_channels, out_channels=cfg.model.cnn_out_channels).to(cfg.device) if cfg.model.cnn_backbone else None

        n_channels, img_height, img_width = calculate_cnn_output_dims(self.cnn_backbone, cfg.data.n_channels, cfg.data.img_height, cfg.data.img_width)

        input_embedder = InputEmbedding(img_height, img_width, cfg.model.patch_size, n_channels, cfg.model.latent_size).to(cfg.device)

        # Vision Transformer model
        self.model = ViT(input_embedder, cfg.model.num_encoders, cfg.model.latent_size, cfg.model.num_heads, cfg.data.num_classes, cfg.model.dropout).to(cfg.device)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.training.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        # TensorBoard logging setup
        self.tensorboard_writer = TensorboardLogger(log_dir=cfg.logger.base_path, step=cfg.logger.step, max_grid_dim=cfg.logger.max_grid_dim)

        # Custom logger
        self.LOGGER = CustomLogger()

    def train_epoch(self, epoch) -> None:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.train()

        # Progress bar for training
        progress_bar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch}/{self.cfg.training.epochs}", position=0, leave=True)

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.cfg.device), labels.to(self.cfg.device)

            n_channels, img_height, img_width = inputs.shape[1], inputs.shape[2], inputs.shape[3]

            self.optimizer.zero_grad()

            if self.cnn_backbone:
                inputs = self.cnn_backbone(inputs)

            outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            # Attention maps processing
            attn_maps = self.model.get_attention_weights()
            attn_maps = processing(reconstruct_attn_from_patches, batch_idx, self.tensorboard_writer.step, attn_maps[1], (img_height, img_width), n_channels, self.cfg.model.patch_size)

            # Compute loss and backpropagation
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Logging training progress
            self.tensorboard_writer.loop_log(progress_bar, len(self.train_loader), epoch, batch_idx, inputs, labels, preds, probs, attn_maps, loss)
            progress_bar.update(1)

        progress_bar.close()

    def validate_epoch(self, epoch: int) -> None:
        """
        Validates the model on the validation dataset.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(total=len(self.test_loader), desc=f"Epoch {epoch}/{self.cfg.training.epochs}", position=0, leave=True)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.cfg.device), labels.to(self.cfg.device)

                if self.cnn_backbone:
                    inputs = self.cnn_backbone(inputs)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                probs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Logging validation progress
                self.tensorboard_writer.loop_log(progress_bar, len(self.test_loader), epoch, batch_idx, inputs, labels, preds, probs, loss)
                progress_bar.update(1)

        avg_loss = running_val_loss / total
        accuracy = correct / total * 100
        self.LOGGER.info(f"Validation Epoch [{epoch}/{self.cfg.training.epochs}]: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        progress_bar.close()

    def train(self) -> None:
        """
        Runs the training loop for the specified number of epochs and validates the model 
        at the end of each epoch.
        """

        if self.weights_path:
            if os.path.exists(self.weights_path):
                self.LOGGER.info(f"Loading model weights from {self.weights_path}")
                self.model.load_state_dict(torch.load(self.weights_path))
            else:
                self.LOGGER.warning(f"Model weights file {self.weights_path} not found. Starting from scratch.")
        
        for epoch in range(1, self.cfg.training.epochs + 1):
            self.LOGGER.info(f"Starting epoch {epoch}/{self.cfg.training.epochs}")
            
            self.train_epoch(epoch)

            # self.validate_epoch(epoch)
            
            # Salvataggio weights
            if epoch % self.cfg.model_checkpoint.save_freq == 0:
                model_save_path = os.path.join(self.cfg.model_checkpoint.base_path, f"model_epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), model_save_path)
                self.LOGGER.info(f"Model weights saved to {model_save_path}")