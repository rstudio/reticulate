

# Here's where I am right now, the goal was to have simple function
# names and "flatten" objects to have a more functional interface:

loss <- reduce_mean(square(y - y_data))
optimizer <- gradient_descent_optimizer(0.5)
train <- optimizer %>% optimzer_minimize(loss)

sess <- session()
sess %>% initialize_all_variables()
for (step in 1:201) {
  sess %>% run(train)
}

# A possible problem with this is that tensorflow functions
# have very special behavior (they are typically going to run
# within the execution graph) so it's useful to explicitly
# call them out within the code. There are also lots and lots
# of nested namespaces that will likely need treatment via
# underscore separators. Here's a variation with underscore
# underscore prefixes that map to the python namespaces:

loss <- tf_reduce_mean(tf_square(y - y_data))
optimizer <- tf_train_gradient_descent_optimizer(0.5)
train <- optimizer %>% tf_train_optimizer_minimize(loss)

sess <- tf_session()
sess %>% tf_initialize_all_variables()
for (step in 1:201) {
  sess %>% tf_run(train)
}

# That variation is basically where the Julia interface sits now,
# except they use dot (".") rather than underscore as Julia actually
# supports nested namespaces similar to Python. Here's what that would
# look like in R:

loss <- tf.reduce_mean(tf.square(y - y_data))
optimizer <- tf.train.gradient_descent_optimizer(0.5)
train <- optimizer %>% tf.train.optimizer_minimize(loss)

sess <- tf.session()
sess %>% tf.initialize_all_variables()
for (step in 1:201) {
  sess %>% tf.run(train)
}

# My reservation about where this sits is that we get into
# some really long expressions when attempting to "flatten"
# the API into a more functional one. For example, our API has:

train <- optimizer %>% tf.train.optimizer_minimize(loss)

# Whereas the equivalent Python API has
train <- optimizer$minimize(loss)


# I fear that in the attempt to be functional we're making
# things more verbose and more difficult to map to the
# copius examples and documentation that will exist in the
# Python universe. Here's a revsion that treats objects as
# R ref classes:

loss <- tf_reduce_mean(tf_square(y - y_data))
optimizer <- tf_train_gradient_descent_optimizer(0.5)
train <- optimizer$minimize(loss)

sess <- tf_session()
sess$initialize_all_variables()
for (step in 1:201) {
  sess$run(train)
}

# As much as I'd like our API to present functional I think
# the sheer scope of namespaces and classes available in the
# TensorFlow API will make this verbose and confusing. Note
# also that the optimzer and session in the example above
# are actually stateful objects that are mutated so it's even
# a bit obscuring to present them functionally.

# If the above variation is reasonable then with one more step
# we end up with an R API that is identical in every way to
# the Python API. We simply captialize/camel-case class names
# and use a dot (".") rather than an underscore:

loss <- tf.reduce_mean(tf.square(y - y_data))
optimizer <- tf.train.GradientDescentOptimizer(0.5)
optimzer$minimize(loss)

sess <- tf.Session()
sess$initialize_all_variables()
for (step in 1:201) {
  sess$run(train)
}

# I think there may be huge benefit to having R code and Python
# code be identical as it will be trivial and robust to read
# Python docs and examples and translate them to R and in general
# easier for the two communities to collaborate.





