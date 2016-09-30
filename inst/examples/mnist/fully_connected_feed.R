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

# minst functions
source("mnist.R")

# input_data
input_data <- tf$contrib$learn$datasets$mnist

# Basic model parameters as external flags.
flags <- tf$app$flags
FLAGS <- flags$FLAGS
flags$DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags$DEFINE_integer('max_steps', 5000L, 'Number of steps to run trainer.')
flags$DEFINE_integer('hidden1', 128L, 'Number of units in hidden layer 1.')
flags$DEFINE_integer('hidden2', 32L, 'Number of units in hidden layer 2.')
flags$DEFINE_integer('batch_size', 100L, 'Batch size. Must divide evenly into the dataset sizes.')
flags$DEFINE_string('train_dir', 'MNIST-data', 'Directory to put the training data.')
flags$DEFINE_boolean('fake_data', FALSE, 'If true, uses fake data for unit testing.')
parse_flags() # parse FLAGS from Rscript command line

# Generate placeholder variables to represent the input tensors.
#
# These placeholders are used as inputs by the rest of the model building
# code and will be fed from the downloaded data in the .run() loop, below.
#
# Args:
#   batch_size: The batch size will be baked into both placeholders.
#
# Returns:
#   placeholders$images: Images placeholder.
#   placeholders$labels: Labels placeholder.
#
placeholder_inputs <- function(batch_size) {

  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images <- tf$placeholder(tf$float32, shape(batch_size, IMAGE_PIXELS))
  labels <- tf$placeholder(tf$int32, shape(batch_size))

  # return both placeholders
  list(images = images, labels = labels)
}

# Fills the feed_dict for training the given step.
#
# A feed_dict takes the form of:
#   feed_dict = dict(
#     <placeholder = <tensor of values to be passed for placeholder>,
#     ....
#   )
#
# Args:
#   data_set: The set of images and labels, from input_data.read_data_sets()
#   images_pl: The images placeholder, from placeholder_inputs().
#   labels_pl: The labels placeholder, from placeholder_inputs().
#
# Returns:
#   feed_dict: The feed dictionary mapping from placeholders to values.
#
fill_feed_dict <- function(data_set, images_pl, labels_pl) {
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  batch <- data_set$next_batch(FLAGS$batch_size, FLAGS$fake_data)
  images_feed <- batch[[1]]
  labels_feed <- batch[[2]]
  dict(
    images_pl = images_feed,
    labels_pl = labels_feed
  )
}

# Runs one evaluation against the full epoch of data.
#
# Args:
#   sess: The session in which the model has been trained.
#   eval_correct: The Tensor that returns the number of correct predictions.
#   images_placeholder: The images placeholder.
#   labels_placeholder: The labels placeholder.
#   data_set: The set of images and labels to evaluate,
#             from input_data.read_data_sets().
#
do_eval <- function(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_set) {
  # And run one epoch of eval.
  true_count <- 0  # Counts the number of correct predictions.
  steps_per_epoch <- data_set$num_examples %/% FLAGS$batch_size
  num_examples <- steps_per_epoch * FLAGS$batch_size
  for (step in 1:steps_per_epoch) {
    feed_dict <- fill_feed_dict(data_set,
                                images_placeholder,
                                labels_placeholder)
    true_count <- true_count + sess$run(eval_correct, feed_dict=feed_dict)
  }

  precision <- true_count / num_examples
  cat(sprintf('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f\n',
              num_examples, true_count, precision))
}


# Train MNIST for a number of steps.

# Get the sets of images and labels for training, validation, and
# test on MNIST.
data_sets <- input_data$read_data_sets(FLAGS$train_dir, FLAGS$fake_data)

# Tell TensorFlow that the model will be built into the default Graph.
with(tf$Graph()$as_default(), {

  # Generate placeholders for the images and labels.
  placeholders <- placeholder_inputs(FLAGS$batch_size)

  # Build a Graph that computes predictions from the inference model.
  logits <- inference(placeholders$images, FLAGS$hidden1, FLAGS$hidden2)

  # Add to the Graph the Ops for loss calculation.
  loss <- loss(logits, placeholders$labels)

  # Add to the Graph the Ops that calculate and apply gradients.
  train_op <- training(loss, FLAGS$learning_rate)

  # Add the Op to compare the logits to the labels during evaluation.
  eval_correct <- evaluation(logits, placeholders$labels)

  # Build the summary Tensor based on the TF collection of Summaries.
  summary <- tf$merge_all_summaries()

  # Add the variable initializer Op.
  init <- tf$initialize_all_variables()

  # Create a saver for writing training checkpoints.
  saver <- tf$train$Saver()

  # Create a session for running Ops on the Graph.
  sess <- tf$Session()

  # Instantiate a SummaryWriter to output summaries and the Graph.
  summary_writer <- tf$train$SummaryWriter(FLAGS$train_dir, sess$graph)

  # And then after everything is built:

  # Run the Op to initialize the variables.
  sess$run(init)

  # Start the training loop.
  for (step in 1:FLAGS$max_steps) {

    start_time <- Sys.time()

    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    feed_dict <- fill_feed_dict(data_sets$train,
                                placeholders$images,
                                placeholders$labels)

    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    values <- sess$run(list(train_op, loss), feed_dict = feed_dict)
    loss_value <- values[[2]]

    duration <- Sys.time() - start_time

    # Write the summaries and print an overview fairly often.
    if (step %% 100 == 0) {
      # Print status to stdout.
      cat(sprintf('Step %d: loss = %.2f (%.3f sec)\n',
                  step, loss_value, duration))
      # Update the events file.
      summary_str <- sess$run(summary, feed_dict=feed_dict)
      summary_writer$add_summary(summary_str, step)
      summary_writer$flush()
    }

    # Save a checkpoint and evaluate the model periodically.
    if ((step + 1) %% 1000 == 0 || (step + 1) == FLAGS$max_steps) {
      checkpoint_file <- file.path(FLAGS$train_dir, 'checkpoint')
      saver$save(sess, checkpoint_file, global_step=step)
      # Evaluate against the training set.
      cat('Training Data Eval:\n')
      do_eval(sess,
              eval_correct,
              placeholders$images,
              placeholders$labels,
              data_sets$train)
      # Evaluate against the validation set.
      cat('Validation Data Eval:\n')
      do_eval(sess,
              eval_correct,
              placeholders$images,
              placeholders$labels,
              data_sets$validation)
      # Evaluate against the test set.
      cat('Test Data Eval:\n')
      do_eval(sess,
              eval_correct,
              placeholders$images,
              placeholders$labels,
              data_sets$test)
    }
  }
})


