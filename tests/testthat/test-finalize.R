test_that("py_finalize() works", {
  # make sure that on R session termination, Python finalizers are run.

  file <- tempfile()

  callr::r(function(file) {
    Sys.setenv("RETICULATE_ENABLE_PYTHON_FINALIZER" = "yes")
    library(reticulate)

    py_run_string(sprintf("
import weakref

class Foo:
  def __init__(self):
    weakref.finalize(self, self.on_finalize)

  def on_finalize(self):
    with open(r'%s', 'a') as f:
      f.write('Foo.finalize ran\\n')

import atexit
def on_exit():
  with open(r'%s', 'a') as f:
    f.write('on_exit finalizer ran\\n')

atexit.register(on_exit)
obj = Foo()
", file, file))

    invisible()
  }, list(file))

  x <- readLines(file)

  # check that the finalizers ran
  expect_contains(x, "Foo.finalize ran")
  expect_contains(x, "on_exit finalizer ran")
})
