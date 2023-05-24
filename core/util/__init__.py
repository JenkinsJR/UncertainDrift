import importlib


def try_import(module):
    try:
        importlib.import_module(module)
    # the imported module requires a dependency outside the current environment
    except ModuleNotFoundError:
        pass


try_import('core.util.geodesic')
try_import('core.util.grid')
try_import('core.util.misc')
try_import('core.util.path')
try_import('core.util.random')
try_import('core.util.model')