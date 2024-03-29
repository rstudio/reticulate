context("iterators")

test_that("Iterators reflect values back", {
  skip_if_no_python()
  rlist <- list("foo", "bar", 42L)
  expect_equal(iterate(test$makeIterator(rlist)), rlist)
})


test_that("Generators reflect values back", {
  skip_if_no_python()
  expect_equal(as.integer(iterate(test$makeGenerator(5))) + 1L, seq(5))
})


test_that("Iterators are drained of their values by iteration", {
  skip_if_no_python()
  iter <- test$makeIterator(c(1:5))
  a <- iterate(iter)
  b <- iterate(iter)
  expect_length(b, 0)
})


test_that("infinite iterators can be accessed with iter_next", {
  skip_if_no_python()

  # create an infinite generator
  main <- py_run_string("
def infinite_generator():
  n = 0
  while True:
    yield n
    n += 1
")
  it <- main$infinite_generator()

  # iterate and stop when i is 10
  while(TRUE) {
    i <- iter_next(it)
    if (i == 10) break
  }
  expect_equal(i, 10)
})

test_that("iter_next returns sentinel value when it completes", {
  skip_if_no_python()
  iter <- test$makeIterator(c(1:5))
  while (TRUE) {
    item <- iter_next(iter, NA)
    if (is.na(item))
      break
  }
  expect_equal(item, NA)
})

test_that("python iterables can be iterated", {
  skip_if_no_python()
  builtins <- import_builtins(convert = FALSE)
  range <- builtins$range(1L, 10L)
  result <- iterate(range)
  expect_length(result, 9)
})

sequence_generator <-function(start, completed = NULL) {
  value <- start
  function() {
    value <<- value + 1L
    gc()
    if (value < 20L)
      value
    else
      completed
  }
}

test_that("generator functions iterate as expected", {
  skip_if_no_python()
  gen <- py_iterator(sequence_generator(10L))
  expect_equal(iterate(gen), 11L:19L)
  expect_identical(iterate(gen), list())
  expect_identical(iterate(gen), list())
  expect_identical(iterate(gen), list())
})


test_that("generator functions support a custom completed value", {
  skip_if_no_python()
  gen <- py_iterator(sequence_generator(10L, completed = NA), completed = NA)
  expect_equal(iterate(gen), 11L:19L)
})

test_that("generator functions are always called on the main thread", {
  skip_if_no_python()
  gen <- py_iterator(sequence_generator(10L))
  expect_equal(test$iterateOnThread(gen), 11L:19L)
})

test_that("can't supply a function with arguments", {
  expect_error(
    py_iterator(function(x) x),
    "no arguments"
  )
})

# test simplify=TRUE for all atomic types
test_that("iterate(simplify=TRUE) for atomic types", {
  # integer
  it <- py_eval("(i for i in (1, 2, 3))")
  expect_identical(iterate(it), c(1L, 2L, 3L))

  # double
  it <- py_eval("(i for i in (1., 2., 3.))")
  expect_identical(iterate(it), c(1, 2, 3))

  # character
  it <- py_eval("(i for i in ('abc', 'def', 'ghi'))")
  expect_identical(iterate(it), c('abc', 'def', 'ghi'))

  # logical
  it <- py_eval("(i for i in (True, False, True))")
  expect_identical(iterate(it), c(TRUE, FALSE, TRUE))

  # complex
  it <- py_eval("(i for i in (1+1j, 2+2j, 3+3j))")
  expect_identical(iterate(it), c(1+1i, 2+2i, 3+3i))

  # if can't simplify, returns a list
  it <- py_eval("(i for i in (1, 2, 3, None))")
  expect_identical(iterate(it), list(1L, 2L, 3L, NULL))

  it <- py_eval("(i for i in (1., 2., 3., None))")
  expect_identical(iterate(it), list(1, 2, 3, NULL))

  it <- py_eval("(i for i in ('abc', 'def', 'ghi', None))")
  expect_identical(iterate(it), list('abc', 'def', 'ghi', NULL))

  it <- py_eval("(i for i in (True, False, True, None))")
  expect_identical(iterate(it), list(TRUE, FALSE, TRUE, NULL))

  it <- py_eval("(i for i in (1+1j, 2+2j, 3+3j, None))")
  expect_identical(iterate(it), list(1+1i, 2+2i, 3+3i, NULL))

  # unconverted python objects prevent simplification
  it <- py_eval("(i for i in (1, 2, object()))")
  res <- iterate(it)
  expect_type(res, "list")
  expect_identical(res[1:2], list(1L, 2L))
  expect_identical(class(res[[3]]), "python.builtin.object")

  # convert=FALSE prevents simplification
  it <- py_eval("(i for i in (1, 2, 3))", convert = FALSE)
  res <- iterate(it)
  expect_type(res, "list")
  for(el in res)
    expect_s3_class(el, "python.builtin.object")

})
