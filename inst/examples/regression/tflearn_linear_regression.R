# Fit a linear regression model using tflearn. Example adapted
# from https://github.com/tflearn/tflearn/blob/master/examples/basics/linear_regression.py
#
# NOTE: If you see errors of the form:
#
#    Error in py_call(attrib, args, keywords) :
#      IndexError: list index out of range
#
# then you likely need to restart your R session and run again;
# see https://github.com/tflearn/tflearn/issues/360 for more details.
library(tensorflow)
tflearn <- import("tflearn")

set.seed(123)
n <- 100
X <- as.numeric(seq(1, n))
Y <- 2*X + 1 + rnorm(n, sd = 20)

input <- tflearn$input_data(shape = shape(NULL))
linear <- tflearn$single_unit(input)
regression <- tflearn$regression(
  linear,
  optimizer = "SGD",
  loss = "mean_square",
  metric = "default",
  learning_rate = 0.01
)

model <- tflearn$DNN(regression)
model$fit(
  X,
  Y,
  n_epoch = 1000L,
  show_metric = TRUE,
  snapshot_epoch = FALSE
)

slope <- model$get_weights(linear$W)
intercept <- model$get_weights(linear$b)

c(intercept, slope)

# Compare to R model fit
r_model <- lm(Y ~ X)
coef(r_model) # similar, but not the same
