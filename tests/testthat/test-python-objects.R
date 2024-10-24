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

  expect_contains(class(List(1,2,3)),
                  c("__main__.List",
                    "collections.abc.Sequence",
                    "python.builtin.list",
                    "python.builtin.object"))

  py_bt_list <- import_builtins()$list
  expect_identical(py_bt_list(List(1, 2, "3")), list(1, 2, "3"))

})


test_that("wrapt.ProxyObject dicts can be converted", {
  skip_if_no_python()
  skip_if(!py_module_available("wrapt"))
  skip_if(py_version() >= "3.13")

  # something similar to tensorflow _DictWrapper() class
  # https://github.com/tensorflow/tensorflow/blob/r2.12/tensorflow/python/trackable/data_structures.py#L784
  Dict <- py_run_string("

import wrapt
class Dict(wrapt.ObjectProxy):
  pass

assert isinstance(Dict({}), dict)

")$Dict

  expect_contains(class(Dict(dict())),
                  c("__main__.Dict",
                    "python.builtin.ObjectProxy",
                    "python.builtin.object"))

  x <- list("abc" = 1:3)
  py_bt_dict <- import_builtins()$dict
  expect_identical(py_bt_dict(Dict(x)), x)

})


test_that("capsules can be freed by other threads", {
  skip_if_no_python()

  free_py_capsule_on_other_thread <- py_run_string("
import threading
capsule = None

def free_py_capsule_on_other_thread():
  def free():
    global capsule
    del capsule
  t = threading.Thread(target=free)
  t.start()
  t.join()

  ", convert = FALSE)$free_py_capsule_on_other_thread

  e <- new.env(parent = emptyenv())
  e_finalized <- FALSE
  reg.finalizer(e, function(e) { e_finalized <<- TRUE })
  py$capsule <- reticulate:::py_capsule(e)
  remove(e)
  gc()

  expect_false(e_finalized)

  expect_no_error({
    # gctorture()

    py_call_impl(free_py_capsule_on_other_thread, NULL, NULL)

    gc()
    # gctorture(FALSE)
  })

  expect_true(e_finalized)

})


test_that("py_to_r() generics are found from R functions called in Python", {
  skip_if_no_python()


  py_new_callback_caller <- py_run_string("
def py_new_callback_caller(callback):
    def callback_caller(*args, **kwargs):
        callback(*args, **kwargs)
    return callback_caller
", local = TRUE)$py_new_callback_caller

  r_callback <- function(x) {
    expect_false(is_py_object(x))
    expect_identical(x, list(a = 42))
    x
  }
  callback_caller <- py_new_callback_caller(r_callback)


  # simple type conversion handled in py_to_r_cpp
  d <- list(a = 42)
  callback_caller(d) # list() -> dict() -> list()

  # conversion via package S3 method
  od <- import("collections", convert = FALSE)$OrderedDict(d)
  callback_caller(od)

  # conversion via user S3 method
  # ideally we would test by just defining `py_to_r.__main__.MyFoo` in the calling
  # env like this:
  #   (function() {
  #     py_to_r.__main__.MyFoo <- function(x) list(a = 42)
  #     callback_caller(new_MyFoo())
  #   })()
  #
  # unfortunately, our ability to infer the userenv needs a little work.
  # we register the S3 method directly as an alternative.
  new_MyFoo <- py_eval("type('MyFoo', (), {})")
  registerS3method("py_to_r", "__main__.MyFoo", function(x) list(a = 42),
                   asNamespace("reticulate"))
  new_MyFoo <- py_eval("type('MyFoo', (), {})")
  callback_caller(new_MyFoo())

})
