import importlib

def reload_lib(lib):
    """
    Reloads the specified libraries.

        Parameters:
        lib (list): A list of libraries (modules) to reload.
    """
    if isinstance(lib, list):
        for l in lib:
            if isinstance(l, type(importlib)):
                importlib.reload(l)
            else:
                print(f"Warning: {l} is not a module and cannot be reloaded.")
    elif isinstance(lib, type(importlib)):
        importlib.reload(lib)
    else:
        print(f"Warning: {lib} is not a module or a list of modules and cannot be reloaded.")

    