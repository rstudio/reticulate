# When running on Windows in RStudio, we need to patch subprocess.Popen
# https://github.com/rstudio/reticulate/issues/1448


def patch_subprocess_Popen():
    from functools import wraps
    import subprocess

    og_Popen__init__ = subprocess.Popen.__init__

    @wraps(subprocess.Popen.__init__)
    def __init__(self, *args, **kwargs):
        if kwargs.get("stdin") is None:
            kwargs["stdin"] = subprocess.DEVNULL
        return og_Popen__init__(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__
