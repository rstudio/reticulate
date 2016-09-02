

library(magrittr)
library(tensorflow)

hello <- constant('Hello, TensorFlow!')
sess = interactive_session()

sess %>% run(hello)

a <- constant(10)
b <- constant(32)

sess %>% run(a + b)

