context("repl_python() magics")

quiet_repl <- function() {
  options("reticulate.repl.quiet" = TRUE)
  sink(nullfile())
}

unquiet_repl <- function() {
  options("reticulate.repl.quiet" = NULL)
  sink()
}

local_quiet_repl <- function(envir = parent.frame()) {
  quiet_repl()
  withr::defer(unquiet_repl(), envir = envir)
}


test_that("%pwd, %cd", {

  owd <- getwd()
  local_quiet_repl()


  expect_output(
    repl_python(input = "%pwd"),
    paste0(">>>  %pwd\n", owd),
    fixed = TRUE)

  expect_error(
    repl_python(input = "%pwd foo"), "no arguments")

  repl_python(input = c(
    "x = %pwd",
    "%cd ..",
    "y = %pwd",
    "%cd -",
    "z = %pwd"
  ))

  expect_equal(py_eval("x"), owd)
  expect_equal(py_eval("y"), dirname(owd))
  expect_equal(py_eval("z"), owd)

  setwd(owd)

})



test_that("%env", {

  local_quiet_repl()

  repl_python(input = c(
    "x = %env FOOVAR",
    "%env FOOVAR baz",
    "y = %env FOOVAR",
    "%env FOOVAR=foo",
    "z = %env FOOVAR"
    ))

  expect_equal(py_eval("x"), "")
  expect_equal(py_eval("y"), "baz")
  expect_equal(py_eval("z"), "foo")
  Sys.unsetenv("FOOVAR")

})

test_that("%system, !", {

  local_quiet_repl()

  repl_python(input = "x = !ls")
  expect_equal(py_eval("x"), system("ls", intern = TRUE))

})


test_that("%pip", {

  local_quiet_repl()

  virtualenv_create("test-pip-repl-magic")

  expect_true(callr::r(function() {
    Sys.unsetenv("RETICULATE_PYTHON")
    library(reticulate)

    use_virtualenv("test-pip-repl-magic", required = TRUE)

    repl_python(input = "%pip install requests")
    import("requests")
    TRUE
  }))

  virtualenv_remove("test-pip-repl-magic")


})


test_that("%conda", {

  local_quiet_repl()

  capture.output({
    conda_create("test-conda-repl-magic")
  })

  expect_true(callr::r(function() {
    Sys.unsetenv("RETICULATE_PYTHON")
    library(reticulate)

    use_condaenv("test-conda-repl-magic", required = TRUE)

    repl_python(input = "%conda install requests")
    import("requests")
    TRUE
  }, stdout = tempfile("conda output")))

  capture.output({
    conda_remove("test-conda-repl-magic")
  })

})
