
tf_import <- function(module = "tensorflow") {
  if (substring(module, 1, 10) != "tensorflow")
    module <- paste("tensorflow", module, sep=".")
  py_import(module)
}
