import importlib

def is_lib(lib):
    if isinstance(lib, type(importlib)):
        importlib.reload(lib)
    else:
        print(f"Warning: {lib} is not a module and cannot be reloaded.")

def reload_lib(lib):
    """
    Reloads the specified libraries.

        Parameters:
        lib (list): A list of libraries (modules) to reload.
    """
    if isinstance(lib, list):
        for l in lib:
            is_lib(l)
    else:
        is_lib(lib)

    