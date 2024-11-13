
# Methods for Populating the Positron/Ark Variables Pane
# These methods primarily delegate to the implementation in the
# Positron python_ipykernel.inspectors python module.

#' @param x Object to get the display value for
#' @param width Maximum expected width. This is just a suggestion, the UI
#'   can stil truncate the string to different widths.
ark_positron_variable_display_value.python.builtin.object <- function(x, ..., width = getOption("width")) {
  .globals$get_positron_variable_inspector(x)$get_display_value(width)[[1L]]
}


#' @param x Object to get the display type for
#' @param include_length Boolean indicating whether to include object length.
ark_positron_variable_display_type.python.builtin.object <- function(x, ..., include_length = TRUE) {
  i <- .globals$get_positron_variable_inspector(x)
  out <-  i$get_display_type()

  if (startsWith(class(x)[1], "python.builtin.")) # display convert value?
    out <- paste("python", out)

  out
}


#' @param x Object to get the variable kind for
ark_positron_variable_kind.python.builtin.object <- function(x, ...) {
  i <- .globals$get_positron_variable_inspector(x)
  i$get_kind()
}


#' @param x Check if `x` has children
ark_positron_variable_has_children.python.builtin.object <- function(x, ...) {
  i <- .globals$get_positron_variable_inspector(x)
  i$has_children()
}

ark_positron_variable_get_children.python.builtin.object <- function(x, ...) {
  # Return an R list of children. The order of children should be
  # stable between repeated calls on the same object. For example:
  i <- .globals$get_positron_variable_inspector(x)

  get_keys_and_children <- .globals$ark_variable_get_keys_and_children
  if (is.null(get_keys_and_children)) {
    get_keys_and_children <- .globals$ark_variable_get_keys_and_children <-
      import("rpytools.ark_variables", convert = FALSE)$get_keys_and_children
  }

  keys_and_children <- iterate(get_keys_and_children(i), simplify = FALSE)
  children <- iterate(keys_and_children[[2L]], simplify = FALSE)
  names(children) <- as.character(py_to_r(keys_and_children[[1L]]))

  children
}

#' @param index An integer > 1, representing the index position of the child in the
#'   list returned by `ark_variable_get_children()`.
#' @param name The name of the child, corresponding to `names(ark_variable_get_children(x))[index]`.
#'   This may be a string or `NULL`. If using the name, it is the method author's responsibility to ensure
#'   the name is a valid, unique accessor. Additionally, if the original name from `ark_variable_get_children()`
#'   was too long, `ark` may discard the name and supply `name = NULL` instead.
ark_positron_variable_get_child_at.python.builtin.object <- function(x, ..., name, index) {
  # cat("name: ", name, "index: ", index, "\n", file = "~/debug.log", append = TRUE)
  # This could be implemented as:
  #   ark_variable_get_children(x)[[index]]
  i <- .globals$get_positron_variable_inspector(x)
  get_child <- .globals$ark_variable_get_child
  if (is.null(get_child)) {
    get_child <- .globals$ark_variable_get_child <-
      import("rpytools.ark_variables", convert = FALSE)$get_child
    }

    get_child(i, index)
  }


ark_positron_variable_display_type.rpytools.ark_variables.ChildrenOverflow <- function(x, ..., include_length = TRUE) {
  ""
}
ark_positron_variable_kind.rpytools.ark_variables.ChildrenOverflow <- function(x, ...) {
  "empty" # other? collection? map? lazy?
}

ark_positron_variable_display_value.rpytools.ark_variables.ChildrenOverflow <- function(x, ..., width = getOption("width")) {
  paste(py_to_r(x$n_remaining), "more values")
}

ark_positron_variable_has_children.rpytools.ark_variables.ChildrenOverflow <- function(x, ...) {
  FALSE
}
