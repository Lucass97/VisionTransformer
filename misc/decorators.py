def singleton(cls):
    """
    A decorator that ensures a class has only one instance.

    Args:
        cls (type): The class to be decorated as a singleton.

    Returns:
        type: The same class, but ensures only one instance is created.
    """
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper
