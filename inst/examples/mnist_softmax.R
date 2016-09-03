# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# Import data
input_data <- py_import("tensorflow.examples.tutorials.mnist.input_data")

# import tensorflow as tf
tf <- py_import("tensorflow")

flags <- tf$app$flags
FLAGS <- flags$FLAGS
flags$DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist <- input_data$read_data_sets(FLAGS$data_dir, one_hot=TRUE)

sess <- tf$InteractiveSession()

# Create the model
x <- tf$placeholder(tf$float32, list(NULL, 784))
W <- tf$Variable(tf$zeros(c(784, 10)))
b <- tf$Variable(tf$zeros(10))
y <- tf$nn$softmax(tf$matmul(x, W) + b)


# Define loss and optimizer
y_ <- tf$placeholder(tf$float32, list(NULL, 10))
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=list(1L)))
train_step <- tf$train$GradientDescentOptimizer(0.5)$minimize(cross_entropy)

# Train
tf$initialize_all_variables()$run()
for (i in 1:1000) {
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  # this line of code is failing with:
  #
  # Error: Cannot interpret feed_dict key as Tensor: The name 'x' looks like
  # an (invalid) Operation name, not a Tensor. Tensor names must be of the form
  # "<op_name>:<output_index>".
  #
  # The problem is that the dictionary we are passing isn't keyed by tensors!
  train_step$run(list(x = batch_xs, y_ = batch_ys))
}


# # Test trained model
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
