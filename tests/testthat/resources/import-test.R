
library(reticulate)
options(reticulate.logModuleLoad = TRUE)
reticulate::py_run_string("from matplotlib import pyplot as plt")
