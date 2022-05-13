context("dict")

test_that("Python dictionaries can be created", {
  skip_if_no_python()
  expect_is(dict(), "python.builtin.dict")
})

test_that("Python dictionaries can be created with py_dict", {
  skip_if_no_python()
  expect_is(py_dict(list("a", "b", "c"), list(1,2,3)), "python.builtin.dict")
})

test_that("Python dictionaries can use python objects as keys", {
  skip_if_no_python()
  py <- import_builtins(convert = FALSE)
  key <- py$int(42)
  expect_error(dict(key = "foo"), NA)
  expect_is(py_dict(list(key), list("foo")), "python.builtin.dict")
})

test_that("Python dictionaries have numeric keys", {
  skip_if_no_python()
  expect_error(dict(`42` = "foo"), NA)
})

test_that("Python dictionaries can include numbers in their keys", {
  skip_if_no_python()
  expect_error(dict(foo42 = "foo"), NA)
})

test_that("Dictionary items can be get / set / removed with py_item APIs", {
  skip_if_no_python()

  d <- dict()
  one <- r_to_py(1)

  py_set_item(d, "apple", one)
  expect_equal(py_get_item(d, "apple"), one)

  py_del_item(d, "apple")
  expect_error(py_get_item(d, "apple"))
  expect_identical(py_get_item(d, "apple", silent = TRUE), NULL)
})

test_that("$, [ operators behave as expected", {
  skip_if_no_python()

  d <- dict(items = 1, apple = 42)

  expect_true(is.function(d$items))
  expect_true(d['items'] == 1)

  expect_true(d$apple == 42)
  expect_true(d['apple'] == 42)

})

test_that("ordered dictionaries with non-string keys can be converted", {
  skip_if_no_python()

  builtins <- import_builtins(convert = FALSE)
  collections <- import("collections", convert = FALSE)

  t <- builtins$tuple(list(42))
  od <- collections$OrderedDict(list())
  od[[t]] <- 42

  result <- py_to_r(od)
  expect_identical(result, list("(42.0,)" = 42))

})
