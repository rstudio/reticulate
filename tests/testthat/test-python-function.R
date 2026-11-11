context("functions")

test_that("Python functions are marshalled as function objects", {
  skip_if_no_python()
  spec <- inspect$getfullargspec(inspect$getclasstree)
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

test_that("Python calls accept empty, positional, and keyword arguments", {
  skip_if_no_python()
  collect <- py_eval("lambda *args, **kwargs: (args, kwargs)")

  empty <- collect()
  expect_identical(empty[[1L]], list())
  expect_length(empty[[2L]], 0L)

  positional <- collect(1L, 2.5)
  expect_identical(positional[[1L]], list(1L, 2.5))
  expect_length(positional[[2L]], 0L)

  expect_identical(collect(x = 1L), list(list(), list(x = 1L)))
  expect_identical(collect(1L, x = 2.5), list(list(1L), list(x = 2.5)))
})
