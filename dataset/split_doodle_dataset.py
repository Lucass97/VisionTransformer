import os
import pandas as pd
import sys

sys.path.append('../')

from dataset import *
from misc.parser import get_doodle_dataset_splitter_args

class DoodleDatasetSplitter:

    MASTER_FILENAME = 'master_doodle_dataframe.csv'
    TRAIN_FILENAME = 'train_doodle_dataframe.csv'
    VAL_FILENAME = 'val_doodle_dataframe.csv'
    TEST_FILENAME = 'test_doodle_dataframe.csv'

    def __init__(self, base_path: str, df_filename: str = MASTER_FILENAME, random_state: int = 42) -> None:
        """
        Initializes the DoodleDatasetSplitter.

        Parameters:
        - base_path: str, the base directory where the datasets are located.
        - df_filename: str, the name of the master CSV file (default: 'master_doodle_dataframe.csv').
        - random_state: int, the random seed used for reproducibility (default: 42).
        """
        master_path = os.path.join(base_path, df_filename)
        self.master = pd.read_csv(master_path)

        self.train_path = os.path.join(base_path, self.TRAIN_FILENAME)
        self.val_path = os.path.join(base_path, self.VAL_FILENAME)
        self.test_path = os.path.join(base_path, self.TEST_FILENAME)

        self.random_state = random_state
        LOGGER.info(f"Initialized DoodleDatasetSplitter with base path: {base_path}")

    def split_dataset(self, train_ratio: float, val_ratio: float) -> None:
        """
        Splits the dataset into train, validation, and test sets based on the provided ratios.

        Parameters:
        - train_ratio: float, the ratio of data to be used for training (e.g., 0.7 for 70%).
        - val_ratio: float, the ratio of data to be used for validation (e.g., 0.2 for 20%).
        """
        if not (train_ratio + val_ratio) <= 1.0:
            LOGGER.error("The sum of train_ratio and val_ratio must be less than or equal to 1.0.")
            return
        LOGGER.info(f"Starting dataset split with train ratio: {train_ratio} and validation ratio: {val_ratio}")

        # Sampling for the training set
        train = self.master.groupby('word', group_keys=False).sample(frac=train_ratio, random_state=self.random_state)
        LOGGER.info(f"Training set created with {len(train)} samples.")

        # Remaining data to extract the validation set
        remaining_data = self.master[~self.master.index.isin(train.index)]

        # Sampling for the validation set
        val = remaining_data.groupby('word', group_keys=False).sample(frac=val_ratio, random_state=self.random_state)
        LOGGER.info(f"Validation set created with {len(val)} samples.")

        # The remaining data becomes the test set
        test = remaining_data[~remaining_data.index.isin(val.index)]
        LOGGER.info(f"Test set created with {len(test)} samples.")

        # Save the resulting splits to CSV files
        train.to_csv(self.train_path, index=False)
        val.to_csv(self.val_path, index=False)
        test.to_csv(self.test_path, index=False)

        LOGGER.info(f"Dataset splits saved to\n\t\t\t\t {self.train_path},\n\t\t\t\t {self.val_path},\n\t\t\t\t {self.test_path}.")


def main() -> None:
    """
    Main function to parse arguments and initiate the dataset splitting process.
    """
    args = get_doodle_dataset_splitter_args(DoodleDatasetSplitter.MASTER_FILENAME)

    splitter = DoodleDatasetSplitter(args.base_path, args.master_csv_name, args.random_state)

    splitter.split_dataset(args.train_ratio, args.val_ratio)

if __name__ == "__main__":
    main()
