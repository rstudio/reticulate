context("objects")

test_that("the length of a Python object can be computed", {
  skip_if_no_python()

  m <- py_eval("[1, 2, 3]", convert = FALSE)
  expect_equal(length(m), 3L)

  x <- py_eval("None", convert = FALSE)
  expect_identical(length(x), 0L)
  expect_identical(py_bool(x), FALSE)
  expect_error(py_len(x), "'NoneType' has no len()")

  x <- py_eval("object()", convert = FALSE)
  expect_identical(length(x), 1L)
  expect_identical(py_bool(x), TRUE)
  expect_error(py_len(x), "'object' has no len()")

})

test_that("python objects with a __setitem__ method can be used", {
  skip_if_no_python()

  library(reticulate)
  py_run_string('
class M:
  def __getitem__(self, k):
    return "M"
')

  m <- py_eval('M()', convert = TRUE)
  expect_equal(m[1], "M")

  m <- py_eval('M()', convert = FALSE)
  expect_equal(m[1], r_to_py("M"))

})



test_that("py_id() returns unique strings; #1216", {
  skip_if_no_python()

  pypy_id <- py_eval("lambda x: str(id(x))")
  o <- py_eval("object()")
  id <- pypy_id(o)
  expect_identical(py_id(o), pypy_id(o))
  expect_identical(py_id(o), id)

  expect_false(py_id(py_eval("object()")) == py_id(py_eval("object()")))
  expect_true(py_id(py_eval("object")) == py_id(py_eval("object")))
})





test_that("subclassed lists can be converted", {
  skip_if_no_python()

  # modeled after tensorflow ListWrapper() class,
  # automatically applied to all keras and tf modules and models
  # which may contain trackable resources (tensors)
  # https://github.com/tensorflow/tensorflow/blob/r2.12/tensorflow/python/trackable/data_structures.py#L452-L456
  List <- py_run_string("
from collections.abc import Sequence
class List(Sequence, list):
  def __init__(self, *args):
    self._storage = list(args)

  def __getitem__(self, x):
    return self._storage[x]

  def __len__(self):
    return len(self._storage)
")$List

  expect_identical(List(1,2,3), list(1,2,3))

})


test_that("wrapt.ProxyObject dicts can be converted", {
  skip_if_no_python()
  skip_if(!py_module_available("wrapt"))

  # something similar to tensorflow _DictWrapper() class
  # https://github.com/tensorflow/tensorflow/blob/r2.12/tensorflow/python/trackable/data_structures.py#L784
  Dict <- py_run_string("

import wrapt
class Dict(wrapt.ObjectProxy):
  pass

assert isinstance(Dict({}), dict)

")$Dict

  expect_identical(Dict(dict()), structure(list(), names = character(0)))
  expect_identical(Dict(list("abc" = as.list(1:3))), list("abc" = as.list(1:3)))
  withr::with_options(c(reticulate.simplify_lists = TRUE), {
    expect_identical(Dict(list("abc" = 1:3)), list("abc" = 1:3))
  })

})
