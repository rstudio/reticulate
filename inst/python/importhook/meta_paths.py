from .finder import hook_finder


class HookMetaPaths(list):
    """
    Custom list that will ensure any items added are wrapped as a "hooked" finder

    This class is made to replace `sys.meta_paths`
    """

    def __init__(self, finders):
        super(HookMetaPaths, self).__init__([hook_finder(f) for f in finders])

    def __setitem__(self, key, val):
        super(HookMetaPaths, self).__setitem__(key, hook_finder(val))
