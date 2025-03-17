import os

from misc.configs import load_config
from misc.logger.logger import CustomLogger
from misc.logger.prints import print_config
from misc.parser import get_train_args

from trainer import Trainer

LOGGER = CustomLogger()


def main() -> None:

    args = get_train_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"The configuration file '{args.config}' does not exist!")
    
    if args.weights and not os.path.exists(args.weights):
        LOGGER.warning(f"Il percorso dei pesi non esiste: {args.weights}")

    configs, experiment_name = load_config(args.config, args.experiment_name)

    LOGGER.info(f"Experiment: {experiment_name}")
    print_config(configs)

    trainer = Trainer(configs, args.weights)

    trainer.train()


if __name__ == "__main__":
    main()
