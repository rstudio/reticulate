

#' Read and evaluate a Python script
#'
#' Evaluate a Python script and make created Python objects available within R.
#' The Python script is sourced within the Python main module, and so any
#' objects defined are made available within Python as well.
#'
#' @inheritParams py_run_file
#'
#' @param envir The environment to assign Python objects into
#'   (for example, `parent.frame()` or `globalenv()`). Specify `NULL` to
#'   not assign Python objects.
#'
#' @export
source_python <- function(file, envir = parent.frame(), convert = TRUE) {
  
  # source the python script (locally so we can track what mutations are
  # made in the file scope)
  dict <- py_run_file(file, local = TRUE, convert = convert)
  
  # replay changes into the python main module
  main <- import_main(convert = FALSE)
  update <- py_to_r(py_get_attr(main$`__dict__`, "update"))
  update(dict)
  
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
