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

  # register a callback auto-flushing Python output as appropriate
  sys <- NULL
  addTaskCallback(function(...) {

    enabled <- getOption("reticulate.autoflush", default = TRUE)
    if (!enabled)
      return(TRUE)

    if (!is_python_initialized())
      return(TRUE)

    sys <- sys %||% import("sys", convert = TRUE)

    if (!is.null(sys$stdout) && !is.null(sys$stdout$flush))
      sys$stdout$flush()

    if (!is.null(sys$stderr) && !is.null(sys$stderr$flush))
      sys$stderr$flush()

    TRUE

  })

  # on macOS, set the OPENBLAS environment variable if possible, as otherwise
  # numpy will complain that we're using the broken Accelerate BLAS
  #
  # https://github.com/numpy/numpy/issues/15947
  #
  # also set OMP_NUM_THREADS to avoid issues with mixing different OpenMP
  # run-times into the same process
  if (Sys.info()[["sysname"]] == "Darwin") {
    openblas <- Sys.getenv("OPENBLAS", unset = NA)
    if (is.na(openblas) && file.exists("/usr/local/opt/openblas")) {
      Sys.setenv(OPENBLAS = "/usr/local/opt/openblas")
      threads <- Sys.getenv("OMP_NUM_THREADS", unset = NA)
      if (is.na(threads))
        Sys.setenv(OMP_NUM_THREADS = "1")
    }
  }

}

.onUnload <- function(libpath) {
  if (is_python_initialized())
    py_finalize()
  py_clear_last_error()
}
