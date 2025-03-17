
from pydantic import BaseModel
from tabulate import tabulate
from misc.logger.logger import CustomLogger

LOGGER = CustomLogger()


def print_config(config: BaseModel) -> None:
    """
    Prints the configuration object (Pydantic model) in a tabular format.

    Args:
        config (BaseModel): The configuration object to be printed.
    """
    config_dict = config.model_dump()
    table = []

    for section, values in config_dict.items():
        if isinstance(values, dict):
            for key, value in values.items():
                table.append([f"{section}.{key}", value])
        else:
            table.append([section, values])

    LOGGER.info("\n" + tabulate(table, headers=["Config Parameter", "Value"], tablefmt="grid"))
