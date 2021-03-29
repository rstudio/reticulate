

#' Read and evaluate a Python script
#'
#' Evaluate a Python script within the Python main module, then make all public
#' (non-module) objects within the main Python module available within the
#' specified R environment.
#'
#' To prevent assignment of objects into R, pass `NULL` for the `envir`
#' parameter.
#'
#' @inheritParams py_run_file
#'
#' @param envir The environment to assign Python objects into (for example,
#'   `parent.frame()` or `globalenv()`). Specify `NULL` to not assign Python
#'   objects.
#'
#' @export
#' @importFrom utils download.file
source_python <- function(file, envir = parent.frame(), convert = TRUE) {

  # Download file content from URL to a local tempory file
  if (!file.exists(file) && isTRUE(grepl("^https?://", file))) {
    tmpfile <- tempfile(fileext = ".py")
    utils::download.file(url = file, destfile = tmpfile, quiet = TRUE)
    file <- tmpfile
    on.exit(unlink(file), add = TRUE)
  }

  # source the python script into the main python module
  py_run_file(file, local = FALSE, convert = convert)
  on.exit(py_flush_output(), add = TRUE)

  # copy objects from the main python module into the specified R environment
  if (!is.null(envir)) {
    main <- import_main(convert = convert)
    main_dict <- py_get_attr(main, "__dict__")
    names <- py_dict_get_keys_as_str(main_dict)
    names <- names[substr(names, 1, 1) != '_']
    Encoding(names) <- "UTF-8"
    for (name in names) {
      value <- main_dict[[name]]
      if (!inherits(value, "python.builtin.module"))
        assign(name, value, envir = envir)
    }
  }

  # return nothing
  invisible(NULL)
}

