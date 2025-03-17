import argparse


def get_train_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Trainer script"
    )

    # Add the parameter for the experiment name
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing the Trainer parameters"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        help="The name of the experiment for logging or saving results"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        help="Path to the pre-trained model weights file"
    )

    return parser.parse_args()


def get_doodle_dataset_splitter_args(default_master_filename: str) -> argparse.Namespace:
    """
    Parses command-line arguments using argparse.

    Returns:
    - argparse.Namespace object containing the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Doodle Dataset Splitter script: A script to split a master dataset into train, validation, and test sets."
    )

    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="The base directory where the master CSV file and output CSV files will be saved."
    )

    parser.add_argument(
        "--master-csv-name",
        type=str,
        required=False,
        default=default_master_filename,
        help="The name of the master CSV file (default: 'master_doodle_dataframe.csv')."
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        required=False,
        default=0.7,
        help="The ratio of data to be used for training (e.g., 0.7 for 70%)."
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        required=False, 
        default=0.2,
        help="The ratio of data to be used for validation (e.g., 0.2 for 20%)."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        required=False,
        default=42,
        help="The random seed for reproducibility (default: 42)."
    )

    return parser.parse_args()
