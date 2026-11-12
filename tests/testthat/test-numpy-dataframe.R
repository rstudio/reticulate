test_that("NumPy arrays are dataframe columns with first-axis rows", {
  skip_if_no_numpy()
  skip_if_not_installed("tibble")
  skip_if_not_installed("dplyr")
  skip_if(getRversion() < "4.3.0")
  np <- import("numpy", convert = FALSE)
  x <- np$arange(24L, dtype = "int8")$reshape(c(3L, 2L, 4L))

  df <- tibble::tibble(id = c(3L, 1L, 2L), frames = x)
  expect_identical(df$frames, x)
  expect_equal(nrow(df), 3L)
  expect_equal(vctrs::vec_size(x), 3L)
  expect_output(print(df), "numpy")

  result <- dplyr::arrange(dplyr::filter(df, id > 1L), id)
  expect_identical(result$id, c(2L, 3L))
  expect_identical(dim(result$frames), c(2L, 2L, 4L))
  expect_identical(py_to_r(result$frames$dtype$name), "int8")
  expect_true(py_to_r(np$array_equal(result$frames, x$take(c(2L, 0L), axis = 0L))))
  expect_identical(dim(x), c(3L, 2L, 4L))
})

test_that("NumPy slicing retains dtypes, dimensions, and strided values", {
  skip_if_no_numpy()
  skip_if_not_installed("vctrs")
  skip_if(getRversion() < "4.3.0")
  np <- import("numpy", convert = FALSE)
  fixtures <- py_run_string("
import numpy as np
arrays = [
    np.arange(24, dtype='int8').reshape(3, 2, 4)[:, :, ::-1],
    np.asfortranarray(np.arange(12, dtype='float32').reshape(3, 4)),
    np.array([2**63, 2**63 + 1, 2**64 - 1], dtype='uint64'),
    np.array(['a', 'longer', 'c']),
    np.array([(1, 2), (3, 4), (5, 6)], dtype=[('x', 'i1'), ('y', 'i1')]),
    np.arange(6, dtype='>i4').reshape(3, 2),
    np.empty((3, 0, 2), dtype='int8'),
]
", local = TRUE, convert = FALSE)

  for (x in iterate(fixtures$arrays, simplify = FALSE)) {
    for (i in list(c(3L, 1L, 3L), 2L, integer())) {
      result <- vctrs::vec_slice(x, i)
      expect_s3_class(result, "numpy.ndarray")
      expect_identical(dim(result), c(length(i), dim(x)[-1L]))
      expect_true(py_to_r(result$dtype == x$dtype))
      expect_true(py_to_r(np$array_equal(result, x$take(np_array(i - 1L), axis = 0L))))
    }
    expect_true(py_to_r(np$array_equal(vctrs::vec_slice(x, -2L),
                                      x$take(c(0L, 2L), axis = 0L))))
  }
})

test_that("NumPy columns bind and recycle without converting their values", {
  skip_if_no_numpy()
  skip_if_not_installed("tibble")
  skip_if_not_installed("dplyr")
  skip_if(getRversion() < "4.3.0")
  np <- import("numpy", convert = FALSE)
  x <- np$arange(12L, dtype = "int8")$reshape(c(3L, 2L, 2L))
  y <- x$take(np_array(0L), axis = 0L)
  df <- tibble::tibble(id = 1:3, frames = y)
  expect_true(py_to_r(np$array_equal(df$frames, np$`repeat`(y, 3L, axis = 0L))))

  result <- dplyr::bind_rows(tibble::tibble(frames = x),
                             tibble::tibble(frames = y))$frames
  expect_true(py_to_r(np$array_equal(result, np$concatenate(list(x, y)))))
  expect_identical(py_to_r(result$dtype$name), "int8")
  expect_identical(dim(vctrs::vec_ptype(x)), c(0L, 2L, 2L))
  expect_error(vctrs::vec_c(x, np$zeros(c(2L, 4L))), "shape|dimension")
  expect_error(vctrs::vec_ptype_common(x, x$astype("float32")), "dtype")

  x <- np$arange(6L, dtype = ">i4")$reshape(c(3L, 2L))
  result <- dplyr::bind_rows(tibble::tibble(frames = x),
                             tibble::tibble(frames = x))$frames
  expect_true(py_to_r(result$dtype == x$dtype))
  expect_identical(py_to_r(result), rbind(py_to_r(x), py_to_r(x)))
})

test_that("NumPy proxies retain their backing arrays and respect list edits", {
  skip_if_no_numpy()
  skip_if_not_installed("vctrs")
  skip_if(getRversion() < "4.3.0")
  np <- import("numpy", convert = FALSE)
  x <- np$arange(6L, dtype = "int8")$reshape(c(3L, 2L))
  proxy <- vctrs::vec_proxy(x)
  expect_type(proxy, "list")
  expect_length(proxy, 3L)
  expect_identical(vctrs::vec_restore(proxy, x), x)
  expect_null(proxy[c(1L, 4L)][[2L]])
  expect_null(proxy[NA_integer_][[1L]])
  gc()
  expect_true(py_to_r(np$array_equal(proxy[[2L]], x$take(np_array(1L), axis = 0L))))

  edited <- proxy
  edited[[1L]] <- proxy[[3L]]
  result <- vctrs::vec_restore(edited, x)
  expect_true(py_to_r(np$array_equal(result, x$take(c(2L, 1L, 2L), axis = 0L))))
  expect_true(py_to_r(np$array_equal(vctrs::vec_restore(proxy, x), x)))

  rm(x, edited, result)
  gc()
  expect_identical(py_to_r(proxy[[2L]]), matrix(2:3, nrow = 1L))
})

test_that("NumPy columns reject scalars and missing rows explicitly", {
  skip_if_no_numpy()
  skip_if_not_installed("vctrs")
  skip_if(getRversion() < "4.3.0")
  np <- import("numpy", convert = FALSE)
  expect_error(vctrs::vec_size(np$array(1L)), "dimension|scalar")
  expect_error(vctrs::vec_slice(np$arange(3L), NA_integer_), "missing")
  expect_error(vctrs::vec_equal(np$arange(3L), np$arange(3L)), "equality")
  expect_error(vctrs::vec_order(np$arange(3L)), "ordering")
  expect_error(vctrs::vec_c(np$arange(3L), np$arange(3L)), "missing")
})

test_that("NumPy dataframe operations do not materialize array values in R", {
  skip_if_no_numpy()
  skip_if_not_installed("tibble")
  skip_if(getRversion() < "4.3.0")
  skip_if_not(capabilities("profmem"))
  np <- import("numpy", convert = FALSE)
  x <- np$zeros(c(8L, 1024L, 1024L), dtype = "int8")
  path <- tempfile()
  on.exit(unlink(path), add = TRUE)
  Rprofmem(path)
  on.exit(Rprofmem(NULL), add = TRUE)
  df <- tibble::tibble(id = 1:8, frames = x)
  result <- vctrs::vec_slice(df, c(8L, 1L))
  Rprofmem(NULL)

  allocations <- readLines(path)
  allocations <- as.double(sub(" .*", "", grep("^[0-9]", allocations, value = TRUE)))
  expect_lt(max(c(0, allocations)), py_to_r(x$nbytes) / 2)
  expect_identical(dim(result$frames), c(2L, 1024L, 1024L))
  expect_identical(py_to_r(result$frames$dtype$name), "int8")
})
