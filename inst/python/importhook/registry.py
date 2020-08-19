import sys

from .logger import logger


class Hooks(set):
    """Custom set that is used to maintain and call a list of hooks"""
    def __call__(self, module):
        for hook in self:
            logger.debug(f'Calling hook {hook.__module__}:{hook.__name__}({module})')
            mod = hook(module)
            # If they modified the module, then use that instead
            if mod is not None:
                module = mod
        return module


class HookRegistry(dict):
    """
    Class used as the registry for import hooks
    """

    def __setitem__(self, key, hook):
        module_name = key or 'ANY_MODULE'
        logger.debug(f'Registering hook {hook.__module__}:{hook.__name__} for {module_name}')

        # Ensure we have a key for this module and it's value is a `Hooks` set
        if key not in self:
            super(HookRegistry, self).__setitem__(key, Hooks())

        # Add our hook to the registry
        self[key].add(hook)

        # Call hook for a module which has already been loaded
        if key is not None and key in sys.modules:
            module = sys.modules[key]
            logger.debug(f'Calling hook {hook.__module__}:{hook.__name__}({module})')

            module = hook(sys.modules[key])
            if module is not None:
                sys.modules[key] = module


# Create our global registry
registry = HookRegistry()
