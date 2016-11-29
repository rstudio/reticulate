library(tensorflow)

train_inds <- 1:100
test_inds <- 100:150

temp_model_dir <- tempdir()

datasets <- tf$contrib$learn$datasets$load_dataset("iris")

# Infer the real-valued feature columns
feature_columns <- tf$contrib$learn$infer_real_valued_columns_from_input(datasets$data)

# Initialize a DNN classifier with hidden units 10, 15, 10 in each layer
classifier <- tf$contrib$learn$DNNClassifier(
  feature_columns = feature_columns,
  hidden_units = c(10L, 15L, 10L),
  n_classes = 3L,
  model_dir = temp_model_dir)

# Train a DNN Classifier
classifier$fit(datasets$data[train_inds, ], datasets$target[train_inds], steps = 100)

# Generate predictiosn on new data
predictions <- classifier$predict(datasets$data[test_inds, ])
accuracy <- sum(predictions == datasets$target[test_inds]) / length(predictions)
print(paste0("The accuracy is ", accuracy))
