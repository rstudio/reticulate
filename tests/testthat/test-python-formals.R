
context("formals")

fmt_formals <- function(f) {
  formatted <- vapply(names(f), function(n) {
    v <- as.character(f[[n]])
    if (is.null(f[[n]])) paste(n, '= NULL')
    else if (nchar(v) == 0) n
    else paste(n, '=', v)
  }, character(1))
  sprintf('(%s)', paste(formatted, collapse = ', '))
}

expect_formals <- function(given, expected) {
  if (is.character(given)) {
    sig <- given
    object <- py_eval(sprintf('lambda %s: None', given))
    given <- py_get_formals(object)
  } else {
    sig <- '?'
  }
  info <- sprintf('(%s)/%s != %s', sig, fmt_formals(given), fmt_formals(expected))
  expect_identical(as.list(given), expected, info)
}

test_that("Python signatures convert properly", {

  skip_if(py_version() < "3.3")

  expect_formals('a', alist(a = ))
  expect_formals('a, b=1', alist(a = , b = NULL))
  expect_formals('a, *, b=1', alist(a = , ... = , b = NULL))
  expect_formals('a, *args, b=1', alist(a = , ... = , b = NULL))
  expect_formals('a, b=1, **kw', alist(a = , b = NULL, ... = ))
  expect_formals('a, *args, **kw', alist(a = , ... = ))
  expect_formals('a, *args, b=1, **kw', alist(a = , ... = , b = NULL))

})

test_that("Errors from e.g. builtins are not propagated", {

  skip_if(py_version() < "3.3")

  print <- import_builtins()$print
  if(py_version() >= "3.11")
    expect_no_error(py_get_formals(print))
  else
    expect_error(py_get_formals(print))
})

test_that("The inspect.Parameter signature converts properly", {

  skip_if(py_version() < "3.3")

  # Parameter.empty usually signifies no default parameter,
  # but for args 3 and 4 here, it *is* the default parameter.
  inspect <- import("inspect", convert = TRUE)
  Parameter <- inspect$Parameter
  fmls <- py_get_formals(Parameter)
  expect_formals(fmls, alist(
    name = , kind = , ... = ,
    default = NULL, annotation = NULL
  ))

})

test_that("Parameters are not matched by prefix", {

  skip_if(py_version() < "3.3")

  f_r <- function(long = NULL, ...) list(long, list(...))
  f_py <- py_eval('lambda long=None, **kw: (long, kw)')
  expect_identical(formals(f_r), py_get_formals(f_py))

  op <- options(warnPartialMatchArgs = FALSE)
  on.exit(options(op))

  # Normal R functions match partially:
  expect_identical(f_r(l = 2L), list(2L, list()))
  # Python functions behave as expected:
  expect_identical(f_py(l = 2L), list(NULL, list(l = 2L)))

})
