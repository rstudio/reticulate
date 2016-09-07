library(tensorflow)

# import tensorflow
tf <- tensorflow()

# import data
input_data <- tensorflow("examples.tutorials.mnist.input_data")
flags <- tf$app$flags
FLAGS <- flags$FLAGS
flags$DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist <- input_data$read_data_sets(FLAGS$data_dir, one_hot=TRUE)

# initialize the session
sess <- tf$Session()

# Create the model
x <- tf$placeholder(tf$float32, list(NULL, 784L))
W <- tf$Variable(tf$zeros(list(784L, 10L)))
b <- tf$Variable(tf$zeros(list(10L)))

y <- tf$nn$softmax(tf$matmul(x, W) + b)

# Define loss and optimizer
y_ <- tf$placeholder(tf$float32, list(NULL, 10L))
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))
train_step <- tf$train$GradientDescentOptimizer(0.5)$minimize(cross_entropy)

# Train
sess$run(tf$initialize_all_variables())
for (i in 1:1000) {
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step,
           feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

# Test trained model
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(accuracy,
         feed_dict = dict(x = mnist$test$images, y_ = mnist$test$labels))




