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
  expect_equal(py_id(py_get_item(d, "apple")), py_id(one))

  py_del_item(d, "apple")
  expect_error(py_get_item(d, "apple"))
  expect_identical(py_get_item(d, "apple", silent = TRUE), NULL)
})

test_that("$, [ operators behave as expected", {
  skip_if_no_python()

  d <- dict(items = 1, apple = 42)

  expect_true(is.function(d$items))
  expect_true(py_bool(d['items'] == 1))

  expect_true(py_bool(d$apple == 42))
  expect_true(py_bool(d['apple'] == 42))

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

test_that("ordered dictionaries can be converted", {
  skip_if_no_python()

  collections <- import("collections", convert = FALSE)
  od <- collections$OrderedDict(list(tuple("a", 1),
                                     tuple("b", 2),
                                     tuple("c", 3)))

  result <- py_eval("lambda x: x")(od) # implicit conversion to R
  expect_identical(result, list(a = 1, b = 2, c = 3))

})

test_that("py_to_r(dict) converts recursively, #1221", {
  skip_if_no_python()
  skip_if_no_numpy()
  skip_if_no_pandas()

  py <- py_run_string('
import numpy as np
import pandas as pd

np.random.seed(6012022)
tools = ["sas", "stata", "spss", "python", "r", "julia"]

random_df = pd.DataFrame({
"tool": np.random.choice(tools, 500),
"int": np.random.randint(1, 15, 500),
"num": np.random.randn(500),
"bool": np.random.choice([True, False], 500),
"date": np.random.choice(pd.date_range("2020-01-01", "2022-06-01"), 500)
})

# LIST OF DATA FRAMES
df_list = [df for i, df in random_df.groupby(["tool"])]

# DICT OF DATA FRAMES
# begining in Pandas 2.0, .groupby() returns the key as tuple(str,), previously, as a str.
df_dict = {i[0] if isinstance(i, tuple) else i: df for i, df in random_df.groupby(["tool"])}
', local = TRUE)

  rdf_list <- py$df_list
  lapply(rdf_list, expect_s3_class, "data.frame")

  rdf_dict <- py$df_dict
  lapply(rdf_list, expect_s3_class, "data.frame")

  for (i in seq_along(rdf_dict)) {
    attr(rdf_dict[[i]], "pandas.index") <- NULL
    attr(rdf_list[[i]], "pandas.index") <- NULL
  }

  expect_identical(rdf_list, unname(rdf_dict))
  expect_identical(sort(names(rdf_dict)),
                   sort(c("sas", "stata", "spss", "python", "r", "julia")))

})
