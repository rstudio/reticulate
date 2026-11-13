context("vectors")

test_that("Single element vectors are treated as scalars", {
  skip_if_no_python()
  expect_true(test$isScalar(5))
  expect_true(test$isScalar(5L))
  expect_true(test$isScalar("5"))
  expect_true(test$isScalar(TRUE))
})

test_that("Multi-element vectors are treated as lists", {
  skip_if_no_python()
  expect_true(test$isList(c(5,5)))
  expect_true(test$isList(c(5L,5L)))
  expect_true(test$isList(c("5", "5")))
  expect_true(test$isList(c(TRUE, TRUE)))
})

test_that("The list function forces single-element vectors to be lists", {
  skip_if_no_python()
  expect_false(test$isScalar(list(5)))
  expect_false(test$isScalar(list(5L)))
  expect_false(test$isScalar(list("5")))
  expect_false(test$isScalar(list(TRUE)))
})

test_that("atomic vectors preserve scalar and list conversion", {
  for (values in list(c(-1L, 2L), c(-Inf, NaN), c(1+2i, -3i),
                      c(TRUE, FALSE), c("\u03bb", ""))) {
    for (n in 0:2) {
      x <- values[seq_len(n)]
      object <- r_to_py(x)
      expect_identical(py_to_r(object), if (n) x else list())
      expect_identical(inherits(object, "python.builtin.list"), n != 1L)
    }
  }
})

test_that("Homogeneous Python lists convert to atomic vectors", {
  skip_if_no_python()
  expect_identical(py_eval("[0, -1, 2147483647]"), c(0L, -1L, 2147483647L))
  expect_identical(py_eval("[0., -1.5, float('inf'), float('nan')]"),
                   c(0, -1.5, Inf, NaN))
  expect_identical(py_eval("[True, False, True]"), c(TRUE, FALSE, TRUE))
  expect_identical(py_eval("[0j, 1+2j, -3j]"), c(0i, 1+2i, -3i))
  expect_identical(py_eval("['', 'x', '\\u00fc']"), c("", "x", "\u00fc"))
  expect_identical(py_eval("[]"), list())
  expect_identical(py_eval("[1, 2.5, False, 3j, 'x']"),
                   list(1L, 2.5, FALSE, 3i, "x"))
})

test_that("large Python integers convert to doubles", {
  skip_if_no_python()

  expect_identical(py_eval("-1"), -1L)
  expect_identical(py_eval("2**31 - 1"), 2147483647L)
  expect_identical(py_eval("-(2**31 - 1)"), -2147483647L)

  cases <- list(
    "2**31" = 2147483648,
    "2**31 + 5" = 2147483653,
    "-(2**31 + 1)" = -2147483649,
    "2**40" = 2^40,
    "2000000000000" = 2e12,
    "2**53 - 1" = 2^53 - 1,
    "2**53" = 2^53,
    "2**53 + 1" = 2^53,
    "-(2**53 + 1)" = -2^53,
    "2**60" = 2^60,
    "2**64 + 7" = 2^64,
    "-(2**64 + 7)" = -2^64
  )
  for (code in names(cases))
    expect_identical(py_eval(code), cases[[code]], info = code)

  expect_identical(py_to_r(py_eval("2**40", convert = FALSE)), 2^40)
})

test_that("integer lists promote to double and preserve element order", {
  skip_if_no_python()

  expect_identical(py_eval("[2**40, -1, 0, -2147483647, 2147483647]"),
                   c(2^40, -1, 0, -2147483647, 2147483647))
  expect_identical(py_eval("[-1, 0, 2**40, -2147483647, 2147483647]"),
                   c(-1, 0, 2^40, -2147483647, 2147483647))
  expect_identical(py_eval("[-1, 0, -2147483647, 2147483647, 2**40, -1]"),
                   c(-1, 0, -2147483647, 2147483647, 2^40, -1))
  expect_identical(py_eval("[-1, -(2**64 + 7), 2**53 + 1]"),
                   c(-1, -2^64, 2^53))

  expect_identical(py_eval("{'a': 2**40, 'b': [1, 2**40]}"),
                   list(a = 2^40, b = c(1, 2^40)))
  expect_identical(py_eval("(2**40, -1)"), list(2^40, -1L))
  expect_identical(py_eval("[2**40, False, 1.5]"), list(2^40, FALSE, 1.5))
  expect_identical(py_eval("[True, 2**40]"), list(TRUE, 2^40))
  expect_identical(py_eval("True"), TRUE)
})

test_that("integers beyond double range raise and clear Python errors", {
  skip_if_no_python()

  for (code in c("10**400", "-(10**400)", "[10**400, 1]",
                 "[-1, 1, 10**400]", "[2**40, -1, 10**400]")) {
    expect_error(py_eval(code), "OverflowError", info = code)
    expect_identical(py_eval("-1"), -1L)
    expect_identical(py_eval("[1, 2**40]"), c(1, 2^40))
  }
})

test_that("integers beyond double range raise in Python when passed to R functions", {
  skip_if_no_python()

  call_r <- py_run_string("
def call_r_with_huge_int(f, keyword):
    events = []
    try:
        f(x=10**400) if keyword else f(10**400)
    except OverflowError:
        events.append('except')
    finally:
        events.append('finally')
    return events
", local = TRUE)$call_r_with_huge_int

  called <- FALSE
  f <- function(x) called <<- TRUE

  expect_identical(call_r(f, keyword = FALSE), c("except", "finally"))
  expect_identical(call_r(f, keyword = TRUE), c("except", "finally"))
  expect_false(called)
})

test_that("Python INT_MIN preserves R missing integers through promotion", {
  skip_if_no_python()

  # R exports NA_integer_ as an ordinary Python INT_MIN. Retain its
  # sentinel meaning, including before and after promotion to double.
  expect_identical(py_eval("-(2**31)"), NA_integer_)
  expect_identical(py_eval("[1, -(2**31), -1]"), c(1L, NA_integer_, -1L))
  expect_identical(py_eval("[1, -(2**31), 2**40]"), c(1, NA_real_, 2^40))
  expect_identical(py_eval("[2**40, -(2**31), -1]"), c(2^40, NA_real_, -1))
  expect_identical(py_to_r(r_to_py(NA_integer_)), NA_integer_)
  expect_identical(py_to_r(r_to_py(c(1L, NA_integer_, -1L))),
                   c(1L, NA_integer_, -1L))
})

test_that("NumPy object arrays use checked Python integer conversion", {
  skip_if_no_numpy()

  value <- py_eval("__import__('numpy').array([1, 2**40, -1], dtype=object)")
  expect_identical(value, array(list(1L, 2^40, -1L)))
})
