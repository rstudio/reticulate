# When running on Windows in RStudio, we need to patch subprocess.Popen
# https://github.com/rstudio/reticulate/issues/1448


def patch_subprocess_Popen():
    from subprocess import Popen, DEVNULL
    from functools import partialmethod

    Popen.__init__ = partialmethod(Popen.__init__, stdin=DEVNULL)
