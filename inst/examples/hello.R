

library(magrittr)
library(tensorflow)

sess <- interactive_session()

a <- constant(10L)
a


b <- constant(32L)
b

x <- constant(matrix(c(1:4), nrow = 2, ncol = 2))

sess %>% run(a + b)

