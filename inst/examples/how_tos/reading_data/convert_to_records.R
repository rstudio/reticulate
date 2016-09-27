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

# Converts MNIST data to TFRecords file format with Example protos.
library(tensorflow)

# input_data
mnist <- tf$contrib$learn$python$learn$datasets$mnist

SOURCE_URL <- 'http://yann.lecun.com/exdb/mnist/'

TRAIN_IMAGES <- 'train-images-idx3-ubyte.gz'  # MNIST filenames
TRAIN_LABELS <- 'train-labels-idx1-ubyte.gz'
TEST_IMAGES <- 't10k-images-idx3-ubyte.gz'
TEST_LABELS <- 't10k-labels-idx1-ubyte.gz'

tf$app$flags$DEFINE_string('directory', '/tmp/data',
  'Directory to download data files and write the converted result')
FLAGS <- tf$app$flags$FLAGS

int64_feature <- function(value) {
  tf$train$Feature(
    int64_list = tf$train$Int64List(value = list(as.integer(value)))
  )
}

bytes_feature <- function(value) {
  tf$train$Feature(
    bytes_list = tf$train$BytesList(value = list(value))
  )
}

convert_to <- function(data_set, name) {
  images <- data_set$images
  images_dim <- dim(images)
  labels <- data_set$labels
  num_examples <- data_set$num_examples

  if (images_dim[[1]] != num_examples) {
    stop(sprintf("Images size %d does not match label size %d",
                 nrow(images), num_examples))
  }

  rows <- images_dim[[2]]
  cols <- images_dim[[3]]
  depth <- images_dim[[4]]

  filename <- file.path(FLAGS$directory, paste0(name, ".tfrecords"))
  cat("Writing ", filename, "\n")
  writer <- tf$python_io$TFRecordWriter(filename)
  for (index in 1:num_examples) {

    # get the image matrix, convert it to a C-aligned vector, and
    # get the raw bytes underlying this vector
    image <- images[index,,,]       # select image
    image <- aperm(image, c(2,1))   # transpose to row-major
    image <- c(image)               # flatten for serialization
    conn <- rawConnection(raw(), open = "r+")
    writeBin(image, conn)
    image_raw <- rawConnectionValue(conn)
    close(conn)

    # create the feature
    feature <- list(
      height = int64_feature(rows),
      width = int64_feature(cols),
      depth = int64_feature(depth),
      label = int64_feature(labels[[index]]),
      image_raw = bytes_feature(image_raw)
    )

    # write it
    example <- tf$train$Example(features = tf$train$Features(feature = feature))
    writer$write(example$SerializeToString())
  }
  writer$close()
}

# Get the data.
data_sets <- mnist$read_data_sets(FLAGS$directory,
                                  dtype = tf$uint8,
                                  reshape = FALSE)

# Convert to Examples and write the result to TFRecords.
convert_to(data_sets$train, "train")
convert_to(data_sets$validation, "validation")
convert_to(data_sets$test, "test")




