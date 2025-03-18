import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DoodleDataset(Dataset):
    """
    A PyTorch Dataset for loading doodle images and their labels.
    
    This dataset reads image paths and labels from a CSV file and applies 
    a series of transformations to prepare the images for training.
    """

    def __init__(self, root_dir, train=True) -> None:
        """
        Initializes the DoodleDataset class, loading images and applying transformations.

        Args:
            root_dir (str): The root directory where the CSV files are located.
            train (bool): Flag to select between training and validation data.
                         Defaults to True (training data).
        """
        csv_filename = 'train_doodle_dataframe.csv' if train else 'val_doodle_dataframe.csv'
        self.csv_path = os.path.join(root_dir, csv_filename)
        self.root_dir = root_dir
        
        self.df = pd.read_csv(self.csv_path)


        # Define the transformation pipeline for grayscale images
        self.transform = transforms.Compose([
            transforms.Resize((254, 254)),  # Resize images to 255x255
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
            transforms.ToTensor(),  # Convert images to tensor format
            transforms.RandomInvert(p=1.0)
        ])


    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx) -> tuple:
        """
        Retrieves the image and label for a given index, applying the necessary transformations.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (Tensor): The transformed image.
                - label (int): The label associated with the image.
        """
        record = self.df.iloc[idx]
        img_path, label, word = record['image_path'], record['label'], record['word']

        img_path = os.path.join(self.root_dir, img_path)

        image = Image.open(img_path).convert('L')  # Open the image and convert to RGB format
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        
        return image, label
