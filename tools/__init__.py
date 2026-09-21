import importlib
import pkgutil
from .register import function_register

# Import every submodule so @function_register decorators run at package load.
def import_all_modules(package):
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)


import_all_modules(__import__(__name__, fromlist=['']))


__all__ = ['function_register'] 