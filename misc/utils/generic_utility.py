import datetime


def generate_experiment_name(dataset_name: str) -> str:
    """
    Generates a unique experiment name using the dataset name and current timestamp.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        str: The experiment name in the format "<dataset_name>_<timestamp>".
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    return experiment_name


def processing(func, batch_idx, step, *args, **kwargs):
    """
    Esegue la funzione `func` solo quando (batch_idx + 1) % step == 0.
    
    :param func: La funzione da eseguire
    :param batch_idx: Indice del batch
    :param step: Passo per il controllo della condizione
    :param args: Argomenti posizionali per `func`
    :param kwargs: Argomenti keyword per `func`
    """
    if (batch_idx + 1) % step == 0:
        return func(*args, **kwargs)
    return None