context("dynamic-dots")

test_that("callables support dynamic-dots", {

  fn <- py_eval("lambda *args, **kwargs: (args if args else None, kwargs if kwargs else None)")

  # splicing (unpacking) arguments works correctly
  args <- list(1, 2, 3)
  kwargs <- list(a = 4, b = 5, c = 6)
  expect_identical(fn(!!!args), fn(1, 2, 3))
  expect_identical(fn(!!!kwargs), fn(a = 4, b = 5, c = 6))
  expect_identical(fn(!!!c(args, kwargs)), fn(1, 2, 3, a = 4, b = 5, c = 6))
  expect_identical(fn(!!!c(args, kwargs), x = 7), fn(1, 2, 3, a = 4, b = 5, c = 6, x = 7))
  expect_identical(fn(8, !!!c(args, kwargs), x = 7), fn(8, 1, 2, 3, a = 4, b = 5, c = 6, x = 7))
  expect_identical(fn(8, !!!c(args, kwargs), x = 7), list(list(8, 1, 2, 3), list(a = 4, b = 5, c = 6, x = 7)))

  # injecting names works correctly
  nm <- "key"
  expect_identical(fn("{nm}" := 42),     list(NULL, list(key = 42)))
  expect_identical(fn("abc_{nm}" := 42), list(NULL, list(abc_key = 42)))

  # trailing commas are ignored
  expect_identical(fn(1, 2, ), fn(1, 2))
})
