import os
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir: str, step: int = 100, max_grid_dim: int = 16) -> None:
        """
        Initializes the TensorBoard logger.

        Args:
            log_dir (str): Directory to save TensorBoard logs.
            train_dim (int): Number of batches per epoch.
            step (int): Interval (in batches) to log metrics (default is 100).
            max_grid_dim (int): Maximum number of images to log per batch (default is 16).
        """
        log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = step
        self.max_grid_dim = max_grid_dim
        self.avg_training_loss = 0.0

    def loop_log(self, progress_bar, dim: int, epoch: int, batch_idx: int,
                 inputs, labels, preds, probs, attn_maps, loss) -> None:
        """
        Logs training metrics including loss, images, labels, and probabilities to TensorBoard.

        Args:
            epoch (int): Current epoch number.
            batch_idx (int): Current batch index.
            inputs (Tensor): Batch of input images.
            labels (Tensor): True labels for the batch.
            preds (Tensor): Predicted labels for the batch.
            probs (Tensor): Output probabilities for each class.
            loss (float): The current batch loss.
        """

        global_step = epoch * dim + batch_idx
        self.avg_training_loss += loss.item()

        if (batch_idx + 1) % self.step == 0:
  
            progress_bar.set_postfix(loss=f"{self.avg_training_loss/self.step:.4f}")
            self.writer.add_scalar("Loss/train", self.avg_training_loss/self.step, global_step)

            # Log input images as a grid
            img_grid = vutils.make_grid(inputs[:self.max_grid_dim], normalize=True)
           
            self.writer.add_image("Input Images", img_grid, global_step)

            if attn_maps is not None:
                attn_grid = vutils.make_grid(attn_maps[:self.max_grid_dim], normalize=True)
                self.writer.add_image("Attention Maps", attn_grid, global_step)

            # Log true/ predicted labels and max probabilities
            for i in range(min(self.max_grid_dim, inputs.size(0))):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                max_prob = probs[i].max().item()
                log_text = f"True: {true_label}, Pred: {pred_label}, Max Prob: {max_prob:.4f}"
                self.writer.add_text(f"Labels_and_Probabilities/Image_{i}", log_text, global_step)

            self.avg_training_loss = 0.0
            