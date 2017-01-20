# Showcase how a simple linear regression might be fit using TensorFlow.
# We'll fit the model:
#
#   Y ~ XW + b
#
# This example was adapted from:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
library(tensorflow)

# 'X' and 'Y' are placeholders for input data, output data.
# We'll fit a regression model using 3 predictors, so our
# input matrix 'X' will have three columns.
p <- 3L
X <- tf$placeholder("float", shape = shape(NULL, p), name = "x-data")
Y <- tf$placeholder("float", name = "y-data")

# Define the weights for each column in X. Since we will
# have 3 predictors, our 'W' matrix of coefficients will
# have 3 elements. We use a 3 x 1 matrix here, rather than
# a vector, to ensure TensorFlow understands how to perform
# matrix multiplication on 'W' and 'X'.
W <- tf$Variable(tf$zeros(list(p, 1L)))
b <- tf$Variable(tf$zeros(list(1L)))

# Define the model (how estimates of 'Y' are produced)
Y_hat <- tf$add(tf$matmul(X, W), b)

# Define the cost function.
# We seek to minimize the mean-squared error.
cost <- tf$reduce_mean(tf$square(Y_hat - Y))

# Define the mechanism used to optimize the loss function.
# Although normally we'd just use ordinary least squares,
# we'll instead use a gradient descent optimizer (since, in
# a more typical learning situation, we won't have an easy
# mechanism for directly computing the values of coefficients)
generator <- tf$train$GradientDescentOptimizer(learning_rate = 0.01)
optimizer <- generator$minimize(cost)

# Initialize a TensorFlow session for our regression.
init <- tf$global_variables_initializer()
session <- tf$Session()
session$run(init)

# Generate some data. The 'true' model will be 'y = 2x + 1';
# that is, the 'slope' parameter is '2', and the intercept is '1'.
# The variable 'y' will only be associated with the first variable;
# the other two variables are just noise.
set.seed(123)
n <- 250
x <- matrix(runif(p * n), nrow = n)
y <- matrix(2 * x[, 1] + 1 + (rnorm(n, sd = 0.25)))

# Repeatedly run optimizer until the cost is no longer changing.
# (We can take advantage of this since we're using gradient descent
# as our optimizer)
feed_dict <- dict(X = x, Y = y)
epsilon <- .Machine$double.eps
last_cost <- Inf
while (TRUE) {
  session$run(optimizer, feed_dict = feed_dict)
  current_cost <- session$run(cost, feed_dict = feed_dict)
  if (last_cost - current_cost < epsilon) break
  last_cost <- current_cost
}

# Generate an R model and compare coefficients from each fit
r_model <- lm(y ~ x)

# Collect coefficients from TensorFlow model fit
tf_coef <- c(session$run(b), session$run(W))
r_coef  <- r_model$coefficients

print(rbind(tf_coef, r_coef))
