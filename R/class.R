inject_super <- function(fun) {
  e <- new.env()
  e$super <- function() {
    bt <- reticulate::import_builtins()
    self <- as.list(parent.frame())$self
    bt$super(self$`__class__`, self)
  }
  environment(e$super) <- environment(fun)
  environment(fun) <- e
  fun
}

#' Create a python class
#' 
#' @param classname The classname
#' @param defs The definitions 
#' @param inherit List of python classes
#'
#' @export
PyClass <- function(classname, defs, inherit = NULL) {
  bt <- reticulate::import_builtins()
  
  if (inherits(inherit, "python.builtin.object"))
    inherit <- list(inherit)
  
  if (is.list(inherit))
    inherit <- do.call(reticulate::tuple, inherit)
  else
    inherit <- tuple()
  
  defs <- lapply(defs, function(x) {
    if (inherits(x, "function")) {
      x <- inject_super(x)
    }
    x
  })
  
  bt$type(
    classname, inherit, 
    do.call(reticulate::dict, defs)
  )
}

