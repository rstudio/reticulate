library(tensorflow)

# Create 100 phony x, y data points, y = x * 0.1 + 0.3
x_data <- runif(100, min=0, max=1)
y_data <- x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W <- variable(random_uniform(1, -1.0, 1.0))
b = variable(zeros(1))
y = W * x_data + b

# Minimize the mean squared errors.
loss <- reduce_mean(square(y - y_data))
train <-
  gradient_descent_optmizer(0.5) %>%
  optimizer_minimize(loss)

# Launch the graph and initialize the variables.
sess = session()
run(sess, initialize_all_variables())

# Fit the line (Learns best fit is W: 0.1, b: 0.3)
for (step in 1:201) {
  run(sess, train)
  if (step %% 20 == 0)
    cat(step, "-", run(sess, W), run(sess, b), "\n")
}


