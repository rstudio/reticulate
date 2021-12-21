test_that("repl_python() magics", {

  owd <- getwd()
  op <- options("reticulate.repl.quiet" = TRUE)
  sink(nullfile())

  on.exit({
    setwd(owd)
    options(op)
    sink()
  })

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

  repl_python(input = "x = !ls")
  expect_equal(py_eval("x"), system("ls", intern = TRUE))

})
