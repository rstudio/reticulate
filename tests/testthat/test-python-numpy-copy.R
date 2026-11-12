test_that("converting compatible NumPy arrays avoids an intermediate buffer", {
  skip_if_no_numpy()
  np <- import("numpy", convert = FALSE)
  skip_if(package_version(py_to_r(np$`__version__`)) < "1.13")
  tracemalloc <- import("tracemalloc", convert = FALSE)
  skip_if(py_to_r(tracemalloc$is_tracing()))

  # 'i' is NumPy's C int dtype, matching R's integer storage on every platform.
  for (dtype in c("float64", "i")) {
    x <- np$ones(c(1024L, 1024L), dtype = dtype, order = "F")
    x$setflags(write = FALSE)
    invisible(py_to_r(x))
    size <- py_to_r(x$nbytes)

    tracemalloc$start()
    on.exit(tracemalloc$stop(), add = TRUE)
    result <- py_to_r(x)
    peak <- py_to_r(tracemalloc$get_traced_memory())[[2L]]
    tracemalloc$stop()

    expect_identical(dim(result), c(1024L, 1024L))
    expect_true(all(result == 1))
    expect_lt(peak, size / 2)
  }
})

test_that("NumPy conversion preserves values across array layouts", {
  skip_if_no_numpy()
  np <- import("numpy", convert = FALSE)
  fixtures <- py_run_string("
def layouts(dtype):
    import numpy as np
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    unaligned = np.ndarray(x.shape, dtype=x.dtype,
                           buffer=bytearray(x.nbytes + 1), offset=1)
    unaligned[:] = x
    readonly = np.asfortranarray(x)
    readonly.setflags(write=False)
    return [x, readonly, x[:, ::-1], x.T,
            np.broadcast_to(x[:1], x.shape), unaligned,
            x.astype(x.dtype.newbyteorder()), x[:0], x[:1, :1].reshape(())]
", local = TRUE, convert = FALSE)

  for (dtype in c("bool", "int16", "i", "uint32", "float32", "float64", "complex128")) {
    for (x in iterate(fixtures$layouts(dtype), simplify = FALSE)) {
      before <- x$copy()
      result <- py_to_r(x)
      expected <- py_to_r(x$ravel(order = "F")$tolist())
      if (length(expected))
        expect_equal(as.vector(result), expected)
      else
        expect_length(result, 0L)
      shape <- unlist(py_to_r(x$shape))
      expect_identical(dim(result), if (length(shape)) shape else 1L)

      if (length(result))
        result[1L] <- 0
      expect_true(py_to_r(np$array_equal(x, before)))
    }
  }
})

test_that("NumPy conversion preserves string widths and object values", {
  skip_if_no_numpy()
  np <- import("numpy", convert = FALSE)
  values <- matrix(c("a", "longer string", "", "\u03bb"), nrow = 2L)

  for (dtype in c("U", "O")) {
    for (order in c("C", "F")) {
      x <- np$array(values, dtype = dtype, order = order)
      x$setflags(write = FALSE)
      expect_identical(py_to_r(x), values)
    }
  }

  bytes <- matrix(c("a", "longer string", "", "z"), nrow = 2L)
  expect_identical(py_to_r(np$array(bytes, dtype = "S", order = "F")), bytes)

  if (package_version(py_to_r(np$`__version__`)) >= "2") {
    x <- np$array(values, dtype = np$dtypes$StringDType(), order = "F")
    x$setflags(write = FALSE)
    expect_identical(py_to_r(x), values)
  }
})

test_that("NumPy datetime arrays retain their units when converted", {
  skip_if_no_numpy()
  np <- import("numpy", convert = FALSE)
  expected <- structure(array(c(0, 86400)), class = c("POSIXct", "POSIXt"))
  for (unit in c("D", "s", "ns")) {
    x <- np$array(c("1970-01-01", "1970-01-02"),
                  dtype = paste0("datetime64[", unit, "]"))
    expect_identical(py_to_r(x), expected)
  }
})

test_that("NumPy object conversion retains a snapshot during callbacks", {
  skip_if_no_numpy()
  objects <- py_run_string("
import numpy as np

class BufferItem:
    __module__ = 'reticulate_test_numpy'
    def __init__(self, value):
        self.value = value

x = np.array([BufferItem(1), BufferItem(2)], dtype=object)
replacement = BufferItem(99)
", local = TRUE, convert = FALSE)

  convert_item <- function(x) {
    value <- py_to_r(x$value)
    if (value == 1L)
      objects$x$`__setitem__`(1L, objects$replacement)
    value
  }
  method <- "py_to_r.reticulate_test_numpy.BufferItem"
  registerS3method("py_to_r", "reticulate_test_numpy.BufferItem", convert_item)
  on.exit(rm(list = method, envir = environment(py_to_r)$.__S3MethodsTable__.),
          add = TRUE)

  expect_identical(py_to_r(objects$x), array(list(1L, 2L)))
  expect_identical(py_to_r(objects$x$`__getitem__`(1L)$value), 99L)
})
