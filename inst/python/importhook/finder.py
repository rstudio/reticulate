import functools

from .loader import HookLoader
from .logger import logger


def hook_finder(finder):
    """
    Helper function to create a new "hooked" subclass of the provided finder class

    This function replaces the `Finder.find_spec` function to ensure that any ModuleSpecs will
    use an `importmod.HookLoader`
    """
    # If this finder has already been 'hooked', then return as-is
    if hasattr(finder, '__hooked__'):
        return finder

    # Determine if we were given an instance or a class
    if isinstance(finder, type):
        finder_cls = finder
    else:
        finder_cls = finder.__class__

    # Determine the class name of the finder
    finder_name = finder_cls.__name__

    def wrap_find_spec(find_spec):
        @functools.wraps(find_spec)
        def wrapper(fullname, path, target=None):
            logger.debug(f'{finder_name}.find_spec(fullname={fullname}, path={path}, target={target})')
            spec = find_spec(fullname, path, target=target)
            if spec is not None:
                spec.loader = HookLoader(spec.loader)
            return spec
        return wrapper

    def wrap_find_loader(find_loader):
        @functools.wraps(find_loader)
        def wrapper(fullname, path):
            logger.debug(f'{finder_name}.find_loader(fullname={fullname}, path={path})')
            loader = find_loader(fullname, path)
            return HookLoader(loader)
        return wrapper

    # Override the functions we care about
    if hasattr(finder, 'find_spec'):
        setattr(finder, 'find_spec', wrap_find_spec(finder.find_spec))
    if hasattr(finder, 'find_loader'):
        setattr(finder, 'find_loader', wrap_find_loader(finder.find_loader))

    # Make this finder as being 'hooked'
    setattr(finder, '__hooked__', True)
    return finder
