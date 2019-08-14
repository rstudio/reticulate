context("functions")

test_that("Python functions are marshalled as function objects", {
  skip_if_no_python()
  spec <- inspect$getargspec(inspect$getclasstree)
  expect_equal(spec$args, c("classes", "unique"))
})

test_that("Python functions can be called by python", {
  skip_if_no_python()
  x <- "foo"
  expect_equal(test$callFunc(test$asString, x), x)
})

test_that("Python callables can be called by R", {
  skip_if_no_python()
  callable <- test$create_callable()
  expect_equal(callable(10), 10)
})

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
    given <- formals(py_eval(sprintf('lambda %s: None', given)))
  } else {
    sig <- '?'
  }
  info <- sprintf('(%s)/%s != %s', sig, fmt_formals(given), fmt_formals(expected))
  expect_identical(as.list(given), expected, info)
}

test_that("Python signatures convert properly", {
  expect_formals('a', alist(a = ))
  expect_formals('a, b=1', alist(a = , b = NULL))
  expect_formals('a, *, b=1', alist(a = , ... = , b = NULL))
  expect_formals('a, *args, b=1', alist(a = , ... = , b = NULL))
  expect_formals('a, b=1, **kw', alist(a = , b = NULL, ... = ))
  expect_formals('a, *args, **kw', alist(a = , ... = ))
  expect_formals('a, *args, b=1, **kw', alist(a = , ... = , b = NULL))
})

test_that("Errors from e.g. builtins are not propagated", {
  print <- import_builtins()$print
  expect_formals(formals(print), alist(... = ))
  expect_match(
    attr(print, 'get_formals_error')$message,
    "ValueError: no signature found for builtin <built-in function print>"
  )
})

test_that("The inspect.Parameter signature converts properly", {
  # Parameter.empty usually signifies no default parameter,
  # but for args 3 and 4 here, it *is* the default parameter.
  Parameter <- import("inspect")$Parameter
  expect_formals(formals(Parameter), alist(
    name = , kind = , ... = ,
    default = NULL, annotation = NULL
  ))
})

test_that("Parameters are not matched by prefix", {
  f_r <- function(long = NULL, ...) list(long, list(...))
  f_py <- py_eval('lambda long=None, **kw: (long, kw)')
  expect_identical(formals(f_r), formals(f_py))

  # Normal R functions match partially:
  expect_identical(f_r(l = 2L), list(2L, list()))
  # Python functions behave as expected:
  expect_identical(f_py(l = 2L), list(NULL, list(l = 2L)))
})
