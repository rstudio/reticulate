args <- commandArgs(TRUE)
venv <- args[[1]]

Sys.unsetenv("RETICULATE_PYTHON")
Sys.unsetenv("RETICULATE_PYTHON_ENV")

reticulate::use_virtualenv(venv, required = TRUE)
sys <- reticulate::import("sys")
cat(sys$path, sep = "\n")
