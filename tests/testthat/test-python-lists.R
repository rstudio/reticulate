context("lists")

# some helpers
test <- import("rpytools.test")

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
  tuple <- test$makeTuple()
  expect_equal(tuple, list(1, 2, 3))
})
