
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

# alias to [[
#' @export
`[[.py_object` <- `$.py_object`

# printing
#' @export
print.py_object <- function(x, ...) {
  py_print(x)
}

# completion
#' @export
.DollarNames.py_object <- function(x, pattern = "") {
  attrs <- py_list_attributes(x)
  attrs[substr(attrs, 1, 1) != '_']
}
