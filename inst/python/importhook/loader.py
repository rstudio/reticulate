from importlib.abc import Loader
import sys

from .logger import logger
from .registry import registry
from .utils import get_module_name


def call_module_hooks(module):
    name = get_module_name(module)

    # If we have a hook in the registry, then call it now
    if name in registry:
        mod = registry[name](module)
        if mod is not None:
            module = mod

    # If we have a global hook in the registry, then call it now
    if None in registry:
        mod = registry[None](module)
        if mod is not None:
            module = mod

    sys.modules[name] = module


class HookLoader(Loader):
    """
    Custom `importlib.abc.Loader` which ensures we call any registered hooks when a module is loaded.
    """
    __slots__ = ['loader']

    def __init__(self, loader):
        self.loader = loader

    def __getattribute__(self, name):
        # If they are requesting the "loader" attribute, return it right away
        loader = super(HookLoader, self).__getattribute__('loader')
        if name == 'loader':
            return loader

        # Pass through attributes/methods only if they exist on the underlying loader
        if hasattr(loader, name):
            try:
                return super(HookLoader, self).__getattribute__(name)
            except AttributeError:
                return getattr(loader, name)

        raise AttributeError

    def create_module(self, *args, **kwargs):
        logger.debug(f'{self.__class__.__name__}.create_module(*args={args}, **kwargs={kwargs})')
        if not hasattr(self.loader, 'create_module'):
            return None

        return self.loader.create_module(*args, **kwargs)

    def find_module(self, name, *args, **kwargs):
        logger.debug(f'{self.__class__.__name__}.find_module(name={name}, *args={args}, **kwargs={kwargs})')
        if not hasattr(self.loader, 'find_module'):
            return None

        module = self.loader.find_module(name=name, *args, **kwargs)
        if module is None:
            return None
        call_module_hooks(module)
        return module

    def load_module(self, name, *args, **kwargs):
        logger.debug(f'{self.__class__.__name__}.load_module(name={name}, *args={args}, **kwargs={kwargs})')
        if not hasattr(self.loader, 'load_module'):
            return None

        module = self.loader.load_module(name, *args, **kwargs)
        if module is None:
            return None
        call_module_hooks(module)
        return module

    def exec_module(self, module, *args, **kwargs):
        logger.debug(f'{self.__class__.__name__}.exec_module(module={module}, *args={args}, **kwargs={kwargs})')
        if not hasattr(self.loader, 'exec_module'):
            return None

        mod = self.loader.exec_module(module, *args, **kwargs)
        if mod is not None:
            module = mod

        call_module_hooks(module)
        return module
