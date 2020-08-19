"""
importhook
==========

Python module for registering hooks to call when certain modules are imported.

.. code:: python

    import importhook

    # Configure a function to call when `socket` is imported
    @importhook.on_import('socket')
    def socket_import(socket):
        print('Socket module imported')

    # Import the `socket` module
    import socket
"""
import functools
import importlib
import sys
import types

from .logger import logger
from .meta_paths import HookMetaPaths
from .registry import registry
from .utils import get_module_name

__all__ = [
    'ANY_MODULE',
    'copy_module',
    'get_module_name',
    'logger',
    'on_import',
    'registry',
    'reload_module',
    'reset_module',
]

ANY_MODULE = None

# Wrap existing (and future) system meta path finders
sys.meta_path = HookMetaPaths(sys.meta_path[:])


def on_import(module_name, func=None):
    """
    Helper function used to register a hook function for a given module

    .. code:: python

        import importhook

        @importhook.on_import('socket')
        def on_socket_import(socket):
            print('socket imported')


        @importhook.on_import(importhook.ANY_MODULE)
        def on_any_import(module):
            print(f'{module.__spec__.name} imported')


        def on_httplib_import(httplib):
            print('httplib imported')


        importhook.on_import('httplib', on_httplib_import)
    """
    if func is None:
        @functools.wraps(func)
        def decorator(func):
            registry[module_name] = func
            return func
        return decorator
    else:
        registry[module_name] = func


def reset_module(module_name):
    """
    Helper function to reset a copied module.

    .. code:: python

        import socket
        import importhook

        # Copy `socket` module
        socket = importhook.copy_module(socket)

        # Reset copied `socket` module back to it's original version
        socket = importhook.reset_module(socket)
    """
    if not isinstance(module_name, str):
        module_name = get_module_name(module_name)

    module = sys.modules.get(module_name)
    if not module:
        return None

    if not hasattr(module, '__original_module__'):
        return module

    sys.modules[module_name] = module.__original_module__
    return module.__original_module__


def copy_module(module, copy_attributes=True, copy_spec=True):
    """
    Helper function for copying a python module

    .. code:: python

        import importhook

        @importhook.on_import('socket')
        def on_socket_import(socket):
            new_socket = importhook.copy_module(socket)
            setattr(new_socket, 'get_hostname', lambda: 'hostname')
            return new_socket
    """
    name = get_module_name(module)
    new_mod = types.ModuleType(name)
    setattr(new_mod, '__original_module__', module)
    setattr(new_mod, '__reset_module__', lambda: reset_module(name))

    # Copy all module attributes
    if copy_attributes:
        for attr, value in module.__dict__.items():
            setattr(new_mod, attr, value)

    # Make a copy of the modules spec if one is present
    if copy_spec and getattr(new_mod, '__spec__', None):
        spec = type(new_mod.__spec__)(name=name, loader=new_mod.__spec__.loader)
        for attr, value in new_mod.__spec__.__dict__.items():
            if attr not in ('name', 'loader'):
                setattr(spec, attr, value)
        new_mod.__spec__ = spec
    return new_mod


def reload_module(module_name):
    """
    Helper function to reload the specified module

    .. code:: python

        import socket
        import importhook

        # Reload the `socket` module by passing in module
        socket = importhook.reload_module(socket)

        # Reload the `socket` module by passing in the name
        socket = importhook.reload_module('socket')
    """
    if not isinstance(module_name, str):
        module_name = get_module_name(module_name)

    module = sys.modules.get(module_name)
    if not module:
        return None

    return importlib.reload(module)
