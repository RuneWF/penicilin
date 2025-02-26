import importlib

def reload_lib(lib):
    """
    Reloads the specified libraries.

        Parameters:
        lib (list): A list of libraries (modules) to reload.
    """
    for l in lib:
        if isinstance(l, type(importlib)):
            importlib.reload(l)
        else:
            print(f"Warning: {l} is not a module and cannot be reloaded.")

    