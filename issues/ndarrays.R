
# https://github.com/rstudio/tensorflow/issues/88

library(reticulate)

a <- array(c(1:24), dim = c(2,3,4))
a

py_a <- r_to_py(a)
py_a
