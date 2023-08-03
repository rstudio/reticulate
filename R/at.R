

#' @rawNamespace if (getRversion() >= "4.3.0") S3method(base::`@`,python.builtin.object)
`@.python.builtin.object` <- function(x, name) {
  if (py_is_module(x) && !py_has_attr(x, name)) {
    module <- py_get_submodule(x, name, py_has_convert(x))
    if (!is.null(module))
      return(module)
  }

  object <- py_get_attr(x, name)
  py_maybe_convert(object, py_has_convert(x))
}

#' @rawNamespace if (getRversion() >= "4.3.0") S3method(base::`@<-`,python.builtin.object)
`@<-.python.builtin.object` <- function(x, name, value) {
  if (!py_is_null_xptr(x) && py_available())
    py_set_attr(x, name, value)
  else
    stop("Unable to assign value (object reference is NULL)")
  x
}

#' @rawNamespace if (getRversion() >= "4.3.0") S3method(utils::.AtNames,python.builtin.module)
.AtNames.python.builtin.module <- function(x, pattern = "") {

  # resolve module proxies (ignore errors since this is occurring during completion)
  result <- tryCatch({
    if (py_is_module_proxy(x))
      py_resolve_module_proxy(x)
    TRUE
  }, error = clear_error_handler(FALSE))
  if (!result)
    return(character())

  # delegate
  .AtNames.python.builtin.object(x, pattern)
}

#' @rawNamespace if (getRversion() >= "4.3.0") S3method(utils::.AtNames,python.builtin.object)
.AtNames.python.builtin.object <- function(x, pattern = "") {

  # skip if this is a NULL xptr
  if (py_is_null_xptr(x) || !py_available())
    return(character())

  # get the names and filter out internal attributes (_*)
  names <- py_suppress_warnings(py_list_attributes(x))

  names <- sort(names, decreasing = FALSE)

  # sort: (name, _name, __name)
  is_dunder <- substr(names, 1, 2) == '__'
  is_oneder <- substr(names, 1, 1) == '_' & !is_dunder
  is_nunder <- substr(names, 1, 1) != '_'
  names <- c(names[is_nunder],
             sprintf("`%s`", names[is_oneder]),
             sprintf("`%s`", names[is_dunder]))

  # replace function with `function`
  names <- sub("^function$", "`function`", names)

  # get the types
  types <- py_suppress_warnings(py_get_attr_types(x, names))

  # if this is a module then add submodules
  if (inherits(x, "python.builtin.module")) {
    name <- py_get_name(x)
    if (!is.null(name)) {
      submodules <- sort(py_list_submodules(name), decreasing = FALSE)
      Encoding(submodules) <- "UTF-8"
      names <- c(names, submodules)
      types <- c(types, rep_len(5L, length(submodules)))
    }
  }

  if(pattern != "") {
    idx <- grepl(pattern, names)
    names <- names[idx]
    types <- types[idx]
  }

  if (length(names) > 0) {
    # set types
    attr(names, "types") <- types

    # specify a help_handler
    attr(names, "helpHandler") <- "reticulate:::help_handler"
  }

  # return
  names
}
