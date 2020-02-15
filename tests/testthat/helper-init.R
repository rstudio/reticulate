
py_tests_initialize <- function() {
  
  # prefer Python 3 if available
  if (is.na(Sys.getenv("RETICULATE_PYTHON", unset = NA))) {
    python <- Sys.which("python3")
    if (nzchar(python))
      use_python(python, required = TRUE)
  }
  
  # import some modules used by the tests
  if (py_available(initialize = TRUE)) {
    
    modules <- list(
      test     = import("rpytools.test"),
      inspect  = import("inspect"),
      sys      = import("sys"),
      builtins = import_builtins(convert = FALSE)
    )
    
    envir <- globalenv()
    list2env(modules, envir = envir)
    
  }
  
}
