
library(magrittr)
library(tensorflow)

hello <- constant('Hello, TensorFlow!')

session() %>%
  run(hello)

