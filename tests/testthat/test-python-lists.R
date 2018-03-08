context("lists")

test_that("R named lists become Python dictionaries", {
  skip_if_no_python()
  l <- list(a = 1, b = 2, c = 3)
  reflected <- test$reflect(l)
  expect_equal(l$a, reflected$a)
  expect_equal(l$b, reflected$b)
  expect_equal(l$c, reflected$c)
})

test_that("Python dictionaries become R named lists", {
  skip_if_no_python()
  l <- list(a = 1, b = 2, c = 3)
  dict <- test$makeDict()
  expect_equal(length(dict), length(l))
  expect_equal(dict$a, l$a)
  expect_equal(dict$b, l$b)
  expect_equal(dict$c, l$c)
})

test_that("R unnamed lists become Python lists", {
  skip_if_no_python()
  l <- list(1L, 2L, 3L)
  expect_equal(test$asString(l), "[1, 2, 3]")
})

test_that("Python unnamed tuples become R unnamed lists", {
  skip_if_no_python()
  l <- list(1, 2, 3)
  tuple1 <- test$makeTuple()
  expect_equal(tuple1, l)

  expect_equal(length(tuple(l)), length(l))
})


test_that("length method for Python lists works", {
  skip_if_no_python()
  py <- import_builtins(convert = FALSE)
  l <- py$list()
  l$append(1)
  l$append(2)
  l$append(3)
  expect_equal(length(l), 3)
})


