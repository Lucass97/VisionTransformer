import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from misc.configs import Config
from misc.logger.logger import CustomLogger
from misc.logger.tensorboard import TensorboardLogger
from misc.utils.generic_utility import processing
from misc.utils.model_utility import reconstruct_attn_from_patches

from dataset.datasets import create_dataset_instance

from models.models import build_model


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

        self.model = build_model(cfg)

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

        self.tensorboard_writer.avg_training_loss = 0.0

        # Progress bar for training
        progress_bar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch}/{self.cfg.training.epochs}", position=0, leave=True)

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            
            inputs, labels = inputs.to(self.cfg.device), labels.to(self.cfg.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            # Attention maps processing
            attn_maps = None
            if self.cfg.model.type == 'vit':
                
                img_height = self.model.input_embedder.img_height
                img_width =self.model.input_embedder.img_width
                
                attn_maps = self.model.get_attention_weights()
                attn_maps = processing(reconstruct_attn_from_patches,
                                       batch_idx, self.tensorboard_writer.step,
                                       attn_maps, (img_height, img_width),
                                       self.cfg.model.patch_size)
            
            # Compute loss and backpropagation
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Logging training progress
            self.tensorboard_writer.loop_train_log(progress_bar,
                                                   len(self.train_loader),
                                                   epoch, batch_idx, inputs,
                                                   labels, preds, probs, attn_maps, loss)
            progress_bar.update(1)

        progress_bar.close()

    def validate_epoch(self, epoch: int) -> None:
        """
        Validates the model on the validation dataset.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        
        self.tensorboard_writer.avg_training_loss = 0.0
        running_val_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(total=len(self.test_loader), desc=f"Epoch {epoch}/{self.cfg.training.epochs}", position=0, leave=True)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.cfg.device), labels.to(self.cfg.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                probs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                # Attention maps processing
                attn_maps = None
                if self.cfg.model.type == 'vit':
                    
                    img_height = self.model.input_embedder.img_height
                    img_width =self.model.input_embedder.img_width
                    
                    attn_maps = self.model.get_attention_weights()
                    attn_maps = processing(reconstruct_attn_from_patches,
                                        batch_idx, self.tensorboard_writer.step,
                                        attn_maps, (img_height, img_width),
                                        self.cfg.model.patch_size)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Logging validation progress
                self.tensorboard_writer.loop_val_log(progress_bar,
                                                     len(self.test_loader),
                                                     epoch, batch_idx, inputs, labels,
                                                     preds, probs, attn_maps, loss)
                progress_bar.update(1)

        avg_loss = running_val_loss / total
        accuracy = correct / total * 100
        self.LOGGER.info(f"Validation Epoch [{epoch}/{self.cfg.training.epochs}]: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        progress_bar.close()

    def train(self) -> None:
        """
        Runs the training loop for the specified number of epochs and validates the model 
        at the end of each epoch. Saves the model even if training is interrupted.
        """

        if self.weights_path:
            if os.path.exists(self.weights_path):
                self.LOGGER.info(f"Loading model weights from {self.weights_path}")
                self.model.load_state_dict(torch.load(self.weights_path))
            else:
                self.LOGGER.warning(f"Model weights file {self.weights_path} not found. Starting from scratch.")

        try:
            for epoch in range(1, self.cfg.training.epochs + 1):
                self.LOGGER.info(f"Starting epoch {epoch}/{self.cfg.training.epochs}")
                
                self.train_epoch(epoch)
                self.validate_epoch(epoch)

                if epoch % self.cfg.model_checkpoint.save_freq == 0:
                    model_save_path = os.path.join(self.cfg.model_checkpoint.base_path, f"model_epoch_{epoch}.pt")
                    torch.save(self.model.state_dict(), model_save_path)
                    self.LOGGER.info(f"Model weights saved to {model_save_path}")

        except KeyboardInterrupt:
            self.LOGGER.warning("Training interrupted by user! Saving model before exiting...")
            interrupt_save_path = os.path.join(self.cfg.model_checkpoint.base_path, f"model_epoch{epoch}_interrupted.pt")
            torch.save(self.model.state_dict(), interrupt_save_path)
            self.LOGGER.info(f"Model weights saved to {interrupt_save_path}")
            return
            
        final_model_path = os.path.join(self.cfg.model_checkpoint.base_path, "model_final.pt")
        torch.save(self.model.state_dict(), final_model_path)
        self.LOGGER.info(f"Final model weights saved to {final_model_path}")
