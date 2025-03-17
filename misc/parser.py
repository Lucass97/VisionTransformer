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
