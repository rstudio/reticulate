context("numpy")

test_that("R matrixes are converted to numpy ndarray", {
  skip_if_no_numpy()
  np <- import("numpy")
  m1 <- matrix(c(1,2,3,4), nrow = 2, ncol = 2)
  m2 <- m1
  expect_equal(np$equal(m1,m2), m1 == m2)
})

test_that("Numpy ndarray is converted to R matrix", {
  skip_if_no_numpy()
  np <- import("numpy")
  m1 <- np$matrix(list(c(1,2), c(3,4)))
  expect_equal(m1, matrix(c(1,2,3,4), nrow = 2, ncol = 2, byrow = TRUE))
})

test_that("Multi-dimensional arrays are handled correctly", {
  skip_if_no_numpy()
  np <- import("numpy")
  a1 <- array(c(1:8), dim = c(2,2,2))
  expect_equal(-a1, np$negative(a1))
})

test_that("Character arrays are handled correctly", {
  skip_if_no_numpy()
  np <- import("numpy")
  a1 <- array(as.character(c(1:8)), dim = c(2,2,2))
  expect_equal(a1, py_to_r(r_to_py(a1)))
})

test_that("Long integer types are converted to R numeric", {
  skip_if_no_numpy()
  np <- import("numpy", convert = FALSE)
  dtypes <- c(np$int64, np$int32,
              np$uint64, np$uint32)
  lapply(dtypes, function(dtype) {
    a1 <- np$array(c(1L:30L), dtype = dtype)
    a1 <- as.vector(py_to_r(a1))
    expect_true(is.numeric(a1))
    expect_equal(a1, 1:30)
  })
})

test_that("Numpy scalars are converted to R vectors", {
  skip_if_no_numpy()
  np <- import("numpy")
  scalar <- c(1.1)
  expect_equal(as.numeric(np$array(scalar)), scalar)
})


test_that("np_array creates arrays of the expected type", {
  skip_if_no_numpy()
  np <- import("numpy")
  a <- np_array(c(1L:10L), dtype = "float32")
  expect_equal(py_to_r(a$dtype$name), "float32")
})

test_that("np_array creates arrays of the expected order", {
  skip_if_no_numpy()
  np <- import("numpy")
  a <- np_array(c(1L:8L), dtype = "float32", order = "F")
  expect_true(py_to_r(a$flags$f_contiguous))
})

test_that("numpy length functions works", {
  skip_if_no_numpy()

  test_array <- function(a) {
    expect_equal(length(a), 8)
  }

  # test no-convert numpy array
  a <- np_array(c(1L:8L), dtype = "float32")
  test_array(a)

  # test convertable numpy array
  np <- import("numpy")
  test_array(r_to_py(matrix(c(1:8), nrow = 2, ncol = 4)))
})

test_that("boolean matrices are converted appropriately", {
  skip_if_no_numpy()

  A <- matrix(TRUE, nrow = 2, ncol = 2)
  expect_equal(A, py_to_r(r_to_py(A)))
})

test_that("numpy string arrays are correctly handled", {
  # https://github.com/rstudio/reticulate/issues/1409
  skip_if_no_numpy()

  np <- import("numpy", convert = FALSE)
  c1 <- np$array(c("1", "2", "3"))
  c2 <- np$array(c("4", "5", "6"))
  x <- py_to_r(np$stack(list(c1, c2), axis = 1L))
  expect_equal(x, matrix(as.character(1:6), ncol = 2))

  # test for more dimensions
  x <- np$random$rand(3L, 5L, 4L)
  y <- x$astype("str")

  a <- py_to_r(x)
  b <- py_to_r(y)
  storage.mode(b) <- "numeric"
  expect_equal(a, b)

  # test F ordering
  x <- np$random$rand(3L, 5L, 4L)
  y <- x$astype("str")
  y <- np$asfortranarray(y)
  expect_true(py_to_r(y$flags$f_contiguous))

  a <- py_to_r(x)
  b <- py_to_r(y)
  storage.mode(b) <- "numeric"
  expect_equal(a, b)

  # test for strided views
  x <- np$random$rand(3L, 5L, 4L)
  y <- x$astype("str")

  x <- np$reshape(x, c(5L, 3L, 4L))
  y <- np$reshape(y, c(5L, 3L, 4L))

  a <- py_to_r(x)
  b <- py_to_r(y)
  storage.mode(b) <- "numeric"
  expect_equal(a, b)

  x <- np$array(as.character(1:18))$reshape(c(-1L, 2L))

  fn <- py_eval("lambda x: x[::2]") # another strided view
  expect_equal(fn(x), matrix(c("1", "2",
                               "5", "6",
                               "9", "10",
                               "13", "14",
                               "17", "18"), byrow = TRUE, ncol = 2))
  x <- np$asfortranarray(x)
  expect_equal(fn(x), matrix(c("1", "2",
                               "5", "6",
                               "9", "10",
                               "13", "14",
                               "17", "18"), byrow = TRUE, ncol = 2))

  })


test_that("numpy non-simple arrays work", {
  # https://github.com/rstudio/reticulate/issues/1613
  py_run_string("import numpy as np", convert = FALSE)
  py_run_string(
    "array = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])",
    convert = FALSE
  )
  result <- py_run_string("rec_array = array.view(np.recarray)", convert = FALSE)

  # Test that attempting to convert a non-simple array fails gracefully,
  # returns a PyObjectRef.
  rec_array <- py_to_r(result$rec_array)
                                      # numpy 1.0 vs 2.0
  expect_s3_class(rec_array, c("numpy.recarray", "numpy.rec.recarray"))
  expect_s3_class(rec_array, "numpy.ndarray")
  expect_s3_class(rec_array, "python.builtin.object")

  skip_if_no_pandas()
  # Test that a registered S3 method for the non-simple numpy array will be
  # called. (Note, some packages, like {zellkonverter}, will register this
  # directly for numpy.ndarray)
  py_to_r.numpy.recarray <- function(x) {
    tryCatch({
      pandas <- import("pandas", convert = FALSE)
      x <- pandas$DataFrame(x)$to_numpy()
      x <- py_to_r(x)
      return(x)
    }, error = identity)
    NextMethod()
  }
  registerS3method("py_to_r", "numpy.recarray", py_to_r.numpy.recarray)
  registerS3method("py_to_r", "numpy.rec.recarray", py_to_r.numpy.recarray)

  on.exit({
    rm(list = c("py_to_r.numpy.recarray", "py_to_r.numpy.rec.recarray"),
       envir = environment(py_to_r)$.__S3MethodsTable__.)
  })

  arr <- py_to_r(result$rec_array)
  expect_identical(arr, rbind(c(1, 2),
                              c(3, 4)))

})

test_that("0-length character arrays are handled correctly", {
  # https://github.com/rstudio/reticulate/issues/1662
  skip_if_no_numpy()
  np <- import("numpy")

  for (dtype in c("U", "U1", "U10", "O", "S", "S1", "S10")) {
    expect_equal(np$array(character(), dtype = dtype),
                 array(character()))
  }
})


test_that("raw arrays convert", {
  skip_if_no_numpy()

  string <- sQuote("Hello World!", "UTF-8")
  rx <- charToRaw(string)

  # roundtrip through bytearray
  expect_identical(rx, py_to_r(r_to_py(rx)))

  # roundtrip through np.array(dtype = "V1")
  rx <- as.array(rx)
  expect_identical(rx, py_to_r(r_to_py(rx)))

  np <- import("numpy", convert = FALSE)
  px <- r_to_py(as.array(rx))

  expect_s3_class(px, "numpy.ndarray")
  expect_equal(py_to_r(px$dtype$name), "void8")
  expect_equal(py_to_r(px$shape), list(18L))
  expect_identical(rx, py_to_r(px))

  # "void" types with itemsize > 1 don't convert.
  px2 <- px$view("V9")
  expect_s3_class(py_to_r(px$view("V9")), "numpy.ndarray")

  # confirm bytes decode to the same string in Python.
  expect_identical(string, py_to_r(px$tobytes()$decode()))

  # run the same tests as above, but w/ numpy<2
  if(py_version() <= "3.11")
    expect_no_error(callr::r(function(python_version) {

      # python_version = "3.10"
      Sys.setenv("RETICULATE_PYTHON" = "managed")
      library(reticulate)
      py_require("numpy<2", python_version)

      string <- sQuote("Hello World!", "UTF-8")
      rx <- charToRaw(string)

      # roundtrip through bytearray
      stopifnot(identical(rx, py_to_r(r_to_py(rx))))

      # roundtrip through np.array(dtype = "V1")
      rx <- as.array(rx)
      stopifnot(identical(rx, py_to_r(r_to_py(rx))))

      np <- import("numpy", convert = FALSE)
      px <- r_to_py(as.array(rx))

      stopifnot(inherits(px, "numpy.ndarray"))
      stopifnot(identical(py_to_r(px$dtype$name), "void8"))
      stopifnot(identical(py_to_r(px$shape), list(18L)))
      stopifnot(identical(rx, py_to_r(px)))

      # "void" types with itemsize > 1 don't convert.
      px2 <- px$view("V9")
      stopifnot(inherits(py_to_r(px$view("V9")), "numpy.ndarray"))

      # confirm bytes decode to the same string in Python.
      stopifnot(identical(string, py_to_r(px$tobytes()$decode())))
    }, list(as.character(py_version()))))

})

test_that("numpy datetime64 values retain units, range, and missing values", {
  skip_if_no_numpy()
  np <- import("numpy", convert = FALSE)

  for (unit in c("Y", "M", "W", "D", "h", "m", "s", "ms", "us")) {
    dates <- np$array(c("2500-01-01", "1600-01-01", "NaT"),
                      dtype = paste0("datetime64[", unit, "]"))
    # Weeks are measured from Thursday, 1970-01-01.
    expected <- as.numeric(as.Date(c("2500-01-01", "1600-01-01")))
    if (unit == "W")
      expected <- floor(expected / 7) * 7
    expected <- c(expected * 86400, NA_real_)
    actual <- py_to_r(dates)
    expect_s3_class(actual, "POSIXct")
    expect_equal(as.numeric(actual), expected)
    for (i in seq_along(expected)) {
      actual <- py_to_r(dates[[i - 1L]])
      expect_s3_class(actual, "POSIXct")
      expect_equal(as.numeric(actual), expected[[i]])
    }
  }

  for (unit in c("ms", "us", "ns", "ps", "fs", "as", "10ms")) {
    dates <- np$array(c(-1L, 0L, 1L), dtype = paste0("datetime64[", unit, "]"))
    scale <- c(ms = 1e-3, us = 1e-6, ns = 1e-9, ps = 1e-12,
               fs = 1e-15, as = 1e-18, "10ms" = 1e-2)[[unit]]
    expect_equal(as.numeric(py_to_r(dates)), c(-1, 0, 1) * scale,
                 tolerance = 1e-20)
    expect_equal(as.numeric(py_to_r(dates[[0L]])), -scale,
                 tolerance = 1e-20)
  }

  expect_identical(as.numeric(py_to_r(np$datetime64("NaT"))), NA_real_)
  near_nat <- np$array("-9223372036854775807", dtype = "int64")$astype("datetime64[ns]")
  expect_true(is.finite(as.numeric(py_to_r(near_nat))))
})

test_that("numpy datetime64 arrays preserve shape and order", {
  skip_if_no_numpy()
  np <- import("numpy", convert = FALSE)
  dates <- np$array(c("2500-01-01", "NaT", "1970-01-01", "1600-01-01"),
                    dtype = "datetime64[D]")$reshape(c(2L, 2L))
  expected <- matrix(as.numeric(as.Date(c("2500-01-01", NA,
                                         "1970-01-01", "1600-01-01"))) * 86400,
                     nrow = 2, byrow = TRUE)
  arrays <- list(dates, np$asfortranarray(dates), dates$T,
                 dates$astype(">M8[D]"))
  for (i in seq_along(arrays)) {
    actual <- py_to_r(arrays[[i]])
    expect_s3_class(actual, "POSIXct")
    expect_identical(dim(actual), c(2L, 2L))
    expect_equal(matrix(as.numeric(actual), nrow = 2),
                 if (i == 3L) t(expected) else expected)
  }
  expect_length(py_to_r(np$array(list(), dtype = "datetime64[D]")), 0L)
  scalar_array <- np$array("2500-01-01", dtype = "datetime64[D]")
  expect_identical(dim(py_to_r(scalar_array)), 1L)
  expect_equal(as.numeric(py_to_r(scalar_array)), expected[[1]])
})
