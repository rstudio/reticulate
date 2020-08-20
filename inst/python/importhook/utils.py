def get_module_name(module):
    """Helper function to get a module's name"""
    if hasattr(module, '__spec__'):
        return module.__spec__.name
    return module.__name__
