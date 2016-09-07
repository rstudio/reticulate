
#' @export
`$.py_object` <- function(x, name) {
  attr <- py_get_attr(x, name)
  if (py_is_callable(attr)) {
    function(...) {
      args <- list()
      keywords <- list()
      dots <- list(...)
      names <- names(dots)
      if (!is.null(names)) {
        for (i in 1:length(dots)) {
          name <- names[[i]]
          if (nzchar(name))
            keywords[[name]] <- dots[[i]]
          else
            args[[length(args) + 1]] <- dots[[i]]
        }
      } else {
        args <- dots
      }
      result = py_call(attr, args, keywords)
      if (is.null(result))
        invisible(result)
      else
        result
    }
  } else {
    py_to_r(attr)
  }
}

#' @export
`[[.py_object` <- `$.py_object`

#' @export
print.py_object <- function(x, ...) {
  py_print(x)
}

#' @export
str.py_object <- function(object, ...) {
  py_str(object)
}

#' @importFrom utils .DollarNames
#' @export
.DollarNames.py_object <- function(x, pattern = "") {

  # get the names and filter out internal attributes (_*)
  names <- py_list_attributes(x)
  names <- names[substr(names, 1, 1) != '_']

  # get the types
  attr(names, "types") <- py_get_attribute_types(x, names)

  # get the doc strings
  inspect <- py_import("inspect")
  attr(names, "docs") <- sapply(names, function(name) {
    inspect$getdoc(py_get_attr(x, name))
  })

  # return
  names
}

#' @export
dict <- function(...) {

  # get the args and their names
  values <- list(...)
  names <- names(values)

  # evaluate names in parent env to get keys
  frame <- parent.frame()
  keys <- lapply(names, function(name) {
    if (exists(name, envir = frame, inherits = TRUE))
      get(name, envir = frame, inherits = TRUE)
    else
      name
  })

  # construct dict
  py_dict(keys, values)
}


# find the name of the python shared library
pythonSharedLibrary <- function() {

  # verify that we have python
  if (!nzchar(Sys.which("python")))
    stop("python not found!")

  # determine version of python
  pythonVersion <- system(intern = TRUE, paste0(
    "python -c 'import sys; print(str(sys.version_info.major) +  \".\" + ",
    " str(sys.version_info.minor))'")
  )

  # add ext and return
  ext <- ifelse(Sys.info()[["sysname"]] == "Darwin", ".dylib", ".so")
  paste0("libpython", pythonVersion, ext)
}
