inject_super <- function(fun) {
  # for each function `fun` we need to place a `super` function in its
  # search path, and this `super` function must be able to access
  # the `self` argument passed to `fun`.

  e <- new.env(parent = environment(fun))

  e$super <- function() {
    bt <- reticulate::import_builtins()
    # self is an argument passed to fun
    self <- get("self", envir = parent.frame(), inherits = FALSE)
    class_ <- get("__class__", envir = e, inherits = FALSE)
    bt$super(class_, self)
  }

  environment(fun) <- e # so fun can access `super`

  fun
}

#' Create a python class
#'
#' @param classname Name of the class. The class name is useful for S3 method
#'  dispatch.
#' @param defs A named list of class definitions - functions, attributes, etc.
#' @param inherit A list of Python class objects. Usually these objects have
#'  the `python.builtin.type` S3 class.
#'
#' @examples
#' \dontrun{
#' Hi <- PyClass("Hi", list(
#'   name = NULL,
#'   `__init__` = function(self, name) {
#'     self$name <- name
#'     NULL
#'   },
#'   say_hi = function(self) {
#'     paste0("Hi ", self$name)
#'   }
#' ))
#'
#' a <- Hi("World")
#' }
#'
#' @export
PyClass <- function(classname, defs = list(), inherit = NULL) {

  builtins <- import_builtins(convert = TRUE)

  if (is_py_object(inherit))
    inherit <- list(inherit)

  bases <- case(

    length(inherit) == 0  ~ tuple(),
    is.list(inherit)      ~ do.call(tuple, inherit),
    is.character(inherit) ~ do.call(tuple, as.list(inherit)),

    ~ stop("unexpected 'inherit' argument")

  )

  defs <- lapply(defs, function(x) {

    # nothing to be done for non-functions
    if (!is.function(x))
      return(x)

    # otherwise, create a new version of the function with 'super' injected
    f <- inject_super(x)

    x <- function(...) {
      # enable conversion scope for `self`
      # the first argument is always `self`.and we don't want to convert it.
      args <- list(...)
      assign("convert", TRUE, envir = as.environment(args[[1]]))
      do.call(f, append(args[1], lapply(args[-1], py_to_r)))
    }

    attr(x, "__env__") <- environment(f)
    x

  })

  type <- builtins$type(
    classname,
    bases,
    do.call(reticulate::dict, defs)
  )

  # we add a reference to the type here. so it can be accessed without needing
  # to find the type from self.
  lapply(defs, function(x) {

    envir <- attr(x, "__env__")
    if (!is.environment(envir))
      return()

    envir$`__class__` <- type

  })

  type

}
