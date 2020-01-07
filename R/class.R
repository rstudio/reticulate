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

#' Enable the conversion scope so `self` fields can be accessed
#' without the need to call `py_to_r`.
#' 
#' @param f a method/function of a Python class
#' 
enable_convert_scope <- function(f) {
  function(...) {
    args <- list(...)
    # enable convertion scope for `self`
    # the first argument is always `self`.and we don't want to convert it.
    assign("convert", TRUE, envir = as.environment(args[[1]])) 
    do.call(f, append(args[1], lapply(args[-1], py_to_r)))
  }
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
  bt <- reticulate::import_builtins()
  
  if (inherits(inherit, "python.builtin.object"))
    inherit <- list(inherit)
  
  if (is.list(inherit))
    inherit <- do.call(reticulate::tuple, inherit)
  else
    inherit <- tuple()
  
  defs <- lapply(defs, function(x) {
    if (inherits(x, "function")) {
      f <- inject_super(x)
      x <- enable_convert_scope(f)
      attr(x, "__env__") <- environment(f)
    }
    x
  })
  
  type <- bt$type(
    classname, inherit, 
    do.call(reticulate::dict, defs)
  )
  
  # we add a reference to the type here. so it can be accessed without needing
  # to find the type from self.
  lapply(defs, function(x) {
    if(!is.null(e <- attr(x, "__env__"))) {
      e$`__class__` <- type
    }
  })
  
  type
}
