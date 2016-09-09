context("lists")

# some helpers
main <- tensorflow:::py_module()
tensorflow:::py_run_string("def asString(x): return str(x)")
tensorflow:::py_run_string("def makeDict(): return {'a': 1.0, 'c': 3.0, 'b': 2.0}")
tensorflow:::py_run_string("def makeTuple(): return (1.0, 2.0, 3.0)")

test_that("R named lists become Python dictionaries", {
  l <- list(a = 1, b = 2, c = 3)
  expect_equal(main$asString(l), "{'a': 1.0, 'c': 3.0, 'b': 2.0}")
})

test_that("R dictionaries become R named lists", {
  l <- list(a = 1, b = 2, c = 3)
  dict <- main$makeDict()
  expect_equal(length(dict), length(l))
  expect_equal(dict$a, l$a)
  expect_equal(dict$b, l$b)
  expect_equal(dict$c, l$c)
})

test_that("R unnamed lists become Python tuples", {
  l <- list(1L, 2L, 3L)
  expect_equal(main$asString(l), "(1, 2, 3)")
})

test_that("R tuples become R unnamed lists", {
  tuple <- main$makeTuple()
  expect_equal(tuple, list(1, 2, 3))
})
