# When running on Windows in RStudio, we need to patch subprocess.Popen
# https://github.com/rstudio/reticulate/issues/1448


def patch_subprocess_Popen():
  import subprocess
  from functools import partial

  subprocess.Popen.__init__ = partial(subprocess.Popen.__init__, stdin = subprocess.DEVNULL)
