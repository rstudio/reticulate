
library(tensorflow)

# imports / aliases
np <- import("numpy")
layers <- tf$contrib$layers
tflearn <- tf$contrib$learn

# Load training dataset
IRIS_TRAINING <- "iris_training.csv"
training_set <- tflearn$datasets$base$load_csv_with_header(
  filename = IRIS_TRAINING,
  target_dtype = np$int,
  features_dtype = np$float32
)

# Load test dataset
IRIS_TEST <- "iris_test.csv"
test_set <- tflearn$datasets$base$load_csv_with_header(
  filename = IRIS_TEST,
  target_dtype = np$int,
  features_dtype = np$float32
)

# Specify that all features have real-value data
feature_columns <- list(layers$real_valued_column("", dimension=4L))

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier <- tflearn$DNNClassifier(
  feature_columns = feature_columns,
  hidden_units = c(10L, 20L, 10L),
  n_classes = 3L,
  model_dir = "/tmp/iris_model"
)

# Fit model.
classifier$fit(x = training_set$data, y = training_set$target, steps = 2000L)

# Evaluate accuracy.
accuracy_score <- classifier$evaluate(
  x = test_set$data,
  y = test_set$target)$accuracy
cat("Accuracy:", format(accuracy_score, digits = 3), "\n")

# Classify two new flower samples.
new_samples <- matrix(c(6.4, 3.2, 4.5, 1.5,
                        5.8, 3.1, 5.0, 1.7),
                      nrow = 2, ncol = 4, byrow = TRUE)
y <- classifier$predict(new_samples)
cat("Predictions:", paste(iterate(y), collapse = ", "))



