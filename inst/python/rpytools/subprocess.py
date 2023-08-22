# When running on Windows in RStudio, we need to patch subprocess.Popen
# https://github.com/rstudio/reticulate/issues/1448


def patch_subprocess_Popen():
  import subprocess
  from functools import partial

  subprocess.Popen = partial(subprocess.Popen, stdin = subprocess.DEVNULL)
