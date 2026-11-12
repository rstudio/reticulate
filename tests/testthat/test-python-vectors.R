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

