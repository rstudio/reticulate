

library(tensorflow)

# import tensorflow
tf <- py_import("tensorflow")

# simple hello world
hello <- tf$constant('Hello, TensorFlow!')
sess <- tf$Session()
sess$run(hello)

# simple add two constants
# TODO: we need to create class specific operator overloads for +, etc.
a <- tf$constant(10)
b <- tf$constant(32)
sess$run(a + b)

# TODO: mnist is likely a namedtuble however we are dropping the names!
flags <- tf$app$flags
flags$DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
FLAGS <- flags$FLAGS
input_data <- py_import("tensorflow.examples.tutorials.mnist.input_data")
mnist <- input_data$read_data_sets(FLAGS$data_dir, one_hot=TRUE)

# TODO: getting a numpy deprecation warning during the build





