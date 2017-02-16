
library(tensorflow)

output <- reticulate::py_capture_output(tf$logging$warn("asdf"))
