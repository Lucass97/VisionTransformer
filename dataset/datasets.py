import importlib

from dataset import *


def create_dataset_instance(full_class_path: str, root_dir: str, train: bool=True):
    '''
    Create an instance of a class dynamically given its full module path.
    
    Args:
        full_class_path (str): The full path of the class, including the module (e.g., 'dataset.dataset_tomato.ClassName').
        root_dir (str): The root directory to pass as an argument to the class constructor.
    
    Returns:
        object: An instance of the specified class.
    
    Raises:
        ModuleNotFoundError: If the module cannot be imported.
        AttributeError: If the class is not found in the specified module.
    '''

    try:
        module_path, class_name = full_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            return getattr(module, class_name)(root_dir, train)
        else:
            LOGGER.error(f"Class '{class_name}' not found in module '{module_path}'")
            raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")
    
    except ModuleNotFoundError as e:
        LOGGER.error(f"Module '{module_path}' not found")
        raise e