#!/usr/bin/env Rscript

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright 2016 RStudio, Inc. All Rights Reserved.
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

library(tensorflow)

flags <- tf$app$flags
flags$DEFINE_boolean('fake_data', FALSE, 'If true, uses fake data for unit testing.')
flags$DEFINE_integer('max_steps', 1000L, 'Number of steps to run trainer.')
flags$DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags$DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags$DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')
FLAGS <- parse_flags()

train <- function() {
  # Import data
  datasets <- tf$contrib$learn$datasets
  mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

  sess <- tf$InteractiveSession()

  # Create a multilayer model.

  # Input placeholders
  with(tf$name_scope("input"), {
    x <- tf$placeholder(tf$float32, shape(NULL, 784L), name = "x-input")
    y_ <- tf$placeholder(tf$float32, shape(NULL, 10L), name = "y-input")
  })

  with(tf$name_scope("input_reshape"), {
    image_shaped_input <- tf$reshape(x, c(-1L, 28L, 28L, 1L))
    tf$image_summary("input", image_shaped_input, 10L)
  })

  # We can't initialize these variables to 0 - the network will get stuck.
  weight_variable <- function(shape) {
    initial <- tf$truncated_normal(shape, stddev = 0.1)
    tf$Variable(initial)
  }

  bias_variable <- function(shape) {
    initial <- tf$constant(0.1, shape = shape)
    tf$Variable(initial)
  }

  # Attach a lot of summaries to a Tensor
  variable_summaries <- function(var, name) {
    with(tf$name_scope("summaries"), {
      mean <- tf$reduce_mean(var)
      tf$scalar_summary(paste0("mean/", name), mean)
      with(tf$name_scope("stddev"), {
        stddev <- tf$sqrt(tf$reduce_mean(tf$square(var - mean)))
      })
      tf$scalar_summary(paste0("stddev/", name), stddev)
      tf$scalar_summary(paste0("max/", name), tf$reduce_max(var))
      tf$scalar_summary(paste0("min/", name), tf$reduce_min(var))
      tf$histogram_summary(name, var)
    })
  }

  # Reusable code for making a simple neural net layer.
  #
  # It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  # It also sets up name scoping so that the resultant graph is easy to read,
  # and adds a number of summary ops.
  #
  nn_layer <- function(input_tensor, input_dim, output_dim,
                       layer_name, act=tf$nn$relu) {
    with(tf$name_scope(layer_name), {
      # This Variable will hold the state of the weights for the layer
      with(tf$name_scope("weights"), {
        weights <- weight_variable(shape(input_dim, output_dim))
        variable_summaries(weights, paste0(layer_name, "/weights"))
      })
      with(tf$name_scope("biases"), {
        biases <- bias_variable(shape(output_dim))
        variable_summaries(biases, paste0(layer_name, "/biases"))
      })
      with (tf$name_scope("Wx_plus_b"), {
        preactivate <- tf$matmul(input_tensor, weights) + biases
        tf$histogram_summary(paste0(layer_name, "/pre_activations"), preactivate)
      })
      activations <- act(preactivate, name = "activation")
      tf$histogram_summary(paste0(layer_name, "/activations"), activations)
    })
    activations
  }

  hidden1 <- nn_layer(x, 784L, 500L, "layer1")

  with(tf$name_scope("dropout"), {
    keep_prob <- tf$placeholder(tf$float32)
    tf$scalar_summary("dropout_keep_probability", keep_prob)
    dropped <- tf$nn$dropout(hidden1, keep_prob)
  })

  y <- nn_layer(dropped, 500L, 10L, "layer2", act = tf$nn$softmax)

  with(tf$name_scope("cross_entropy"), {
    diff <- y_ * tf$log(y)
    with(tf$name_scope("total"), {
      cross_entropy <- -tf$reduce_mean(diff)
    })
    tf$scalar_summary("cross entropy", cross_entropy)
  })

  with(tf$name_scope("train"), {
    optimizer <- tf$train$AdamOptimizer(FLAGS$learning_rate)
    train_step <- optimizer$minimize(cross_entropy)
  })

  with(tf$name_scope("accuracy"), {
    with(tf$name_scope("correct_prediction"), {
      correct_prediction <- tf$equal(tf$arg_max(y, 1L), tf$arg_max(y_, 1L))
    })
    with(tf$name_scope("accuracy"), {
      accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
    })
    tf$scalar_summary("accuracy", accuracy)
  })

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged <- tf$merge_all_summaries()
  train_writer <- tf$train$SummaryWriter(file.path(FLAGS$summaries_dir, "train"),
                                         sess$graph)
  test_writer <- tf$train$SummaryWriter(file.path(FLAGS$summaries_dir, "test"))
  tf$initialize_all_variables()$run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  # Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
  feed_dict <- function(train) {
    if (train || FLAGS$fake_data) {
      batch <- mnist$train$next_batch(100L, fake_data = FLAGS$fake_data)
      xs <- batch[[1]]
      ys <- batch[[2]]
      k <- FLAGS$dropout
    } else {
      xs <- mnist$test$images
      ys <- mnist$test$labels
      k <- 1.0
    }
    dict(x = xs,
         y_ = ys,
         keep_prob = k)
  }

  for (i in 1:FLAGS$max_steps) {
    if (i %% 10 == 0) { # Record summaries and test-set accuracy
      result <- sess$run(list(merged, accuracy), feed_dict = feed_dict(FALSE))
      summary <- result[[1]]
      acc <- result[[2]]
      test_writer$add_summary(summary, i)
    } else {  # Record train set summaries, and train
      if (i %% 100 == 99) { # Record execution stats
        run_options <- tf$RunOptions(trace_level = tf$RunOptions()$FULL_TRACE)
        run_metadata <- tf$RunMetadata()
        result <- sess$run(list(merged, train_step),
                           feed_dict = feed_dict(TRUE),
                           options = run_options,
                           run_metadata = run_metadata)
        summary <- result[[1]]
        train_writer$add_run_metadata(run_metadata, sprintf("step%03d", i))
        train_writer$add_summary(summary, i)
        cat("Adding run metadata for ", i, "\n")
      } else {  # Record a summary
        result <- sess$run(list(merged, train_step), feed_dict = feed_dict(TRUE))
        summary <- result[[1]]
        train_writer$add_summary(summary, i)
      }
    }
  }

  train_writer$close()
  test_writer$close()
}

# initialize summaries_dir (remove existing if necessary)
if (tf$gfile$Exists(FLAGS$summaries_dir))
  tf$gfile$DeleteRecursively(FLAGS$summaries_dir)
tf$gfile$MakeDirs(FLAGS$summaries_dir)

# train
train()





