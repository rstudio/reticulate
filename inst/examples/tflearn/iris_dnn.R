library(tensorflow)


datasets <- tf$contrib$learn$datasets$load_dataset("iris")
feature_columns <- tf$contrib$learn$infer_real_valued_columns_from_input(datasets$data)
classifier <- tf$contrib$learn$DNNClassifier(feature_columns = feature_columns, hidden_units = c(10, 20, 10), n_classes = 3)
classifier$fit(datasets$data, datasets$target, steps = 100)
