
library(reticulate)
options(reticulate.logModuleLoad = TRUE)
py_require("matplotlib")
reticulate::py_run_string("from matplotlib import pyplot as plt")
