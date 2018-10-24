args <- commandArgs(TRUE)
venv <- args[[1]]

reticulate::use_virtualenv(venv, required = TRUE)
sys <- reticulate::import("sys")
cat(sys$path, sep = "\n")
