#' Interact with the Python Main Module
#' 
#' The `py` object provides a means for interacting
#' with the Python main session directly from \R. Python
#' objects accessed through `py` are automatically converted
#' into \R objects, and can be used with any other \R
#' functions as needed.
#'
#' @format An \R object acting as an interface to the
#'   Python main module.
#' 
#' @export
"py"

.onLoad <- function(libname, pkgname) {
  main <- NULL
  makeActiveBinding("py", env = asNamespace(pkgname), function() {
    
    # return main module if already initialized
    if (!is.null(main))
      return(main)
    
    # attempt to initialize main
    if (is_python_initialized())
      main <<- import_main(convert = TRUE)
    
    # return value of main
    main
    
  })
}

.onUnload <- function(libpath) {
  if (is_python_initialized())
    py_finalize()
}
