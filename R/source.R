

#' Read and evaluate a Python script
#'
#' Evaluate a Python script and make created Python objects available within R.
#'
#' @inheritParams py_run_file
#'
#' @param envir The environment to assign Python objects into
#'   (for example, `parent.frame()` or `globalenv()`). Specify `NULL` to
#'   not assign Python objects.
#'
#' @export
source_python <- function(file, envir = parent.frame(), convert = TRUE) {
  
  # source the python script
  dict <- py_run_file(file, local = TRUE, convert = convert)
  
  # get the keys
  names <- py_dict_get_keys_as_str(dict)
  Encoding(names) <- "UTF-8"
  
  # assign the objects into the specified environment
  if (!is.null(envir)) {
    for (name in names)
      assign(name, dict[[name]], envir = envir)
  }
  
  # return nothing
  invisible(NULL)
}
