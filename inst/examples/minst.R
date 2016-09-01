

library(tensorflow)

# import tensorflow
tf <- tf_import()

# simple hello world
hello <- tf$constant('Hello, TensorFlow!')
sess <- tf$Session()
sess$run(hello)

# simple add two constants
# TODO: we need to create class specific operator overloads for +, etc.
a <- tf$constant(10)
b <- tf$constant(32)
c <- a + b
sess$run(a + b)

flags <- tf$app$flags
flags$DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
FLAGS <- flags$FLAGS
input_data <- tf_import("examples.tutorials.mnist.input_data")
mnist <- input_data$read_data_sets(FLAGS$data_dir, one_hot=TRUE)

# TODO: getting a numpy deprecation warning during the build





