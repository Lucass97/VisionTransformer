import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def get_images_path(root_dir: str):
    """
    Retrieves image file paths and their corresponding class labels from the directory structure.

    Args:
        root_dir (str): The root directory where the dataset is stored. Each subdirectory represents 
                        a different class (disease).

    Returns:
        tuple: A tuple containing two lists:
            - image_paths (list): A list of file paths to images.
            - labels (list): A list of class labels corresponding to the images.
    """
    image_paths = []
    labels = []
    
    # Iterate through each subfolder (representing classes)
    for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Iterate through each file in the class folder
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(('.JPG', '.jpg', '.jpeg')):
                file_path = os.path.join(class_path, file_name)
                image_paths.append(file_path)
                labels.append(idx)
    
    return image_paths, labels


class TomatoDataset(Dataset):

    def __init__(self, root_dir, train=True) -> None:
        """
        Initializes the TomatoDataset class, loading images and applying transformations.

        Args:
            root_dir (str): The root directory where the 'train' or 'val' directories are located.
            train (bool): Flag to select between the training and validation data. Defaults to True (training data).
        """
        if train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')

        self.image_paths = []
        self.labels = []

        # Get image paths and labels
        self.image_paths, self.labels = get_images_path(root_dir=self.root_dir)

        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize images to 256x256
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
            transforms.RandomRotation(degrees=15),  # Randomly rotate images
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Apply color jitter
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize images to 224x224
            transforms.ToTensor(),  # Convert images to tensor format
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_paths)

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
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB format
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]  # Get the corresponding label
        
        return image, label
