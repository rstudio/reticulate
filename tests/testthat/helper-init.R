
# prefer Python 3 if available
if (!py_available(initialize = FALSE) &&
    is.na(Sys.getenv("RETICULATE_PYTHON", unset = NA)))
{
  python <- Sys.which("python3")
  if (nzchar(python))
    use_python(python, required = TRUE)
}

print(py_config())

test_that <- function(desc, code) {

  # don't run tests on CRAN
  testthat::skip_on_cran()
  
  # skip if we don't have python
  skip_if_no_python()
  
  # delegate to testthat
  call <- sys.call()
  call[[1L]] <- quote(testthat::test_that)
  eval(call, envir = parent.frame())
  
}

context <- function(label) {
  
  # import some modules used by the tests
  if (py_available(initialize = TRUE)) {
    
    modules <- list(
      test     = import("rpytools.test"),
      inspect  = import("inspect"),
      sys      = import("sys"),
      builtins = import_builtins(convert = FALSE)
    )
    
    list2env(modules, envir = parent.frame())
    
  }
  
  # delegate to testthat
  call <- sys.call()
  call[[1L]] <- quote(testthat::context)
  eval(call, envir = parent.frame())
  
}
