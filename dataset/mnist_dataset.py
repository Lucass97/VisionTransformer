from torchvision import datasets
import torchvision.transforms as transforms


class MNISTDataset():

    def __init__(self, root_dir, train=True) -> None:
        """
        Initializes the MNISTDataset class and applies transformations to the dataset.

        Args:
            root_dir (str): The root directory where the MNIST dataset will be stored or loaded from.
            train (bool): A flag indicating whether to load the training set (True) or test set (False). Defaults to True.
        """
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize images to 32x32
            transforms.RandomRotation(degrees=15),  # Randomly rotate images within a range of 15 degrees
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Randomly crop and resize images to 28x28
            transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
            transforms.ToTensor(),  # Convert images to tensor format
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images with mean and std
        ])

        # Load the MNIST dataset
        self.dataset = datasets.MNIST(
            root=root_dir,
            train=train,
            download=True,
            transform=self.transform
        )
    
    def __new__(cls, *args, **kwargs):
        """
        Creates an instance of the class and returns the MNIST dataset object directly.

        Args:
            *args: Variable length argument list for initialization.
            **kwargs: Keyword arguments for initialization.

        Returns:
            dataset (torchvision.datasets.MNIST): The MNIST dataset with the applied transformations.
        """
        instance = super(MNISTDataset, cls).__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance.dataset
