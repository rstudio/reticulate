test_that("NumPy columns have R row indexing in tibbles and data frames", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("tibble")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)

  for (shape in list(3L, c(3L, 2L, 4L))) {
    x <- np$arange(as.integer(prod(shape)), dtype = "int8")$reshape(shape)
    column <- np_array_column(x)
    expect_length(column, 3L)
    expect_equal(vctrs::vec_size(column), 3L)
    expect_null(dim(column))
    expect_null(names(column))
    expect_error(vctrs::vec_slice(column, "shape"), "name|character|subscript")
    expect_true(py_to_r(np$array_equal(r_to_py(column), x)))
    expect_true(py_to_r(np$array_equal(column, x)))

    for (df in list(tibble::tibble(id = 1:3, pixels = column),
                    data.frame(id = 1:3, pixels = column))) {
      out <- as.data.frame(df)[c(3L, 1L), , drop = FALSE]
      expect_identical(out$id, c(3L, 1L))
      expect_true(py_to_r(np$array_equal(np_array(out$pixels),
                                        x$take(c(2L, 0L), axis = 0L))))
    }
    expect_true(py_to_r(np$array_equal(np_array(column[-2L]),
                                      x$take(c(0L, 2L), axis = 0L))))
    expect_identical(dim(np_array(column[integer()])), c(0L, shape[-1L]))
  }

  # Opting into column semantics must not change ordinary Python objects.
  x <- np$arange(3L)
  expect_false(vctrs::vec_is(x))
  expect_equal(py_to_r(x[0L]), 0L)
  expect_error(np_array_column(np$array(1L)), "dimension|scalar")
  masked <- np$ma$array(1:3, mask = c(FALSE, TRUE, FALSE))
  expect_error(np_array_column(masked), "subclass|ndarray")
  expect_error(np_array_column(x$view(np$recarray)), "subclass|ndarray")
})

test_that("NumPy columns retain values and dtypes across layouts and binding", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("vctrs", "0.7.0")
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
    column <- np_array_column(x)
    for (i in list(c(3L, 1L, 3L), 2L, integer())) {
      result <- np_array(vctrs::vec_slice(column, i))
      expect_identical(dim(result), c(length(i), dim(x)[-1L]))
      expect_true(py_to_r(result$dtype == x$dtype))
      expect_true(py_to_r(np$array_equal(result, x$take(np_array(i - 1L), axis = 0L))))
    }

    bound <- vctrs::vec_c(column[c(3L, 1L)], column[2L])
    result <- np_array(bound)
    expect_true(py_to_r(result$dtype == x$dtype))
    expect_true(py_to_r(np$array_equal(result, x$take(c(2L, 0L, 1L), axis = 0L))))
    expect_identical(dim(np_array(vctrs::vec_ptype(column))), c(0L, dim(x)[-1L]))
  }

  x <- np_array_column(np$zeros(c(3L, 2L), dtype = "int8"))
  expect_error(vctrs::vec_c(x, np_array_column(np$zeros(c(3L, 4L), dtype = "int8"))),
               "shape|dimension")
  expect_error(vctrs::vec_c(x, np_array_column(np$zeros(c(3L, 2L), dtype = "float32"))),
               "dtype")
})

test_that("grouped NumPy transforms recombine rows in their original order", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("tibble")
  skip_if_not_installed("dplyr")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  x <- np$arange(24L, dtype = "int8")$reshape(c(6L, 2L, 2L))
  df <- tibble::tibble(id = 1:6, group = c(2L, 1L, 2L, 1L, 2L, 1L),
                       pixels = np_array_column(x))
  grouped <- dplyr::group_by(df, group)
  result <- dplyr::mutate(grouped, pixels = pixels * 2L)
  expect_identical(result$id, df$id)
  expect_true(py_to_r(np$array_equal(np_array(result$pixels), x * 2L)))

  # Each group returns a one-row wrapper, which dplyr combines with vec_c().
  result <- dplyr::summarise(grouped, pixels = pixels[1L] * 2L)
  expect_identical(result$group, c(1L, 2L))
  expect_true(py_to_r(np$array_equal(np_array(result$pixels),
                                    x$take(c(1L, 0L), axis = 0L) * 2L)))

  combined <- dplyr::bind_rows(df[c(6L, 2L), ], df[c(1L, 4L), ])
  expect_identical(combined$id, c(6L, 2L, 1L, 4L))
  expect_true(py_to_r(np$array_equal(np_array(combined$pixels),
                                    x$take(c(5L, 1L, 0L, 3L), axis = 0L))))
})

test_that("missing NumPy rows remain representable until array conversion", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("tibble")
  skip_if_not_installed("dplyr")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  x <- np$arange(6L, dtype = "uint8")$reshape(c(3L, 2L))
  column <- np_array_column(x)
  df <- tibble::tibble(id = 1:3, pixels = column)
  out <- dplyr::left_join(tibble::tibble(id = c(3L, 1L, 4L)), df, by = "id")
  expect_identical(is.na(out$pixels), c(FALSE, FALSE, TRUE))
  expect_error(np_array(out$pixels), "missing")
  expect_true(py_to_r(np$array_equal(np_array(out$pixels[1:2]),
                                    x$take(c(2L, 0L), axis = 0L))))

  initialized <- vctrs::vec_init(column, 3L)
  expect_true(all(is.na(initialized)))
  initialized <- vctrs::vec_assign(initialized, 1:3, column[c(3L, 1L, 2L)])
  expect_false(anyNA(initialized))
  expect_true(py_to_r(np$array_equal(np_array(initialized),
                                    x$take(c(2L, 0L, 1L), axis = 0L))))
})

test_that("NumPy column arithmetic handles both operands and unary operators", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  x <- np$arange(6L, dtype = "int16")$reshape(c(3L, 2L))
  column <- np_array_column(x)

  results <- list(-column, 2L + column, 100L - column,
                   column + column[c(3L, 1L, 2L)])
  expected <- list(-x, 2L + x, 100L - x,
                    x + x$take(c(2L, 0L, 1L), axis = 0L))
  for (i in seq_along(results)) {
    array <- np_array(results[[i]])
    expect_true(py_to_r(np$array_equal(array, expected[[i]])))
    expect_true(py_to_r(array$dtype == expected[[i]]$dtype))
  }

  # Rows are observations, so they must not broadcast as trailing features.
  matrix_column <- np_array_column(np$ones(c(2L, 3L), dtype = "int16"))
  scalar_rows <- np_array_column(np$arange(3L, dtype = "int16"))
  expect_error(matrix_column + scalar_rows, "trailing dimensions")
  expect_error(matrix_column + np_array_column(np$ones(c(3L, 3L))), "recycle|size")
  expect_true(py_to_r(np$array_equal(np_array(matrix_column + matrix_column[1L]),
                                    np$full(c(2L, 3L), 2L))))
})

test_that("interleaved NumPy sources retain row order and dtype", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  a <- np$arange(6L, dtype = ">i4")$reshape(c(3L, 2L))
  b <- np$arange(10L, 16L, dtype = ">i4")$reshape(c(3L, 2L))
  combined <- vctrs::vec_c(np_array_column(a), np_array_column(b))
  i <- c(4L, 2L, 6L, 1L, 5L, 3L, 4L)
  result <- np_array(combined[i])
  expected <- np$concatenate(list(a, b))$take(i - 1L, axis = 0L)
  expect_true(py_to_r(np$array_equal(result, expected)))
  expect_true(py_to_r(result$dtype == a$dtype))
})

test_that("NumPy column selections retain source reference semantics", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  x <- np$arange(6L, dtype = "int16")$reshape(c(3L, 2L))
  column <- np_array_column(x)
  selected <- column[c(3L, 1L)]
  gathered <- r_to_py(selected)

  x[0L, 0L] <- 99L
  expect_equal(py_to_r(np_array(column)[0L, 0L]), 99L)
  expect_equal(py_to_r(np_array(selected)[1L, 0L]), 99L)
  expect_equal(py_to_r(gathered[1L, 0L]), 0L)

  edited <- vctrs::vec_assign(column, 1L, column[3L])
  expect_equal(py_to_r(np_array(edited)[0L, 0L]), 4L)
  expect_equal(py_to_r(np_array(column)[0L, 0L]), 99L)
})

test_that("empty NumPy columns and complete replacement release source arrays", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  weakref <- import("weakref", convert = FALSE)

  for (empty in list(vctrs::vec_ptype, function(x) x[integer()])) {
    state <- local({
      array <- np$ones(c(3L, 2L), dtype = "int8")
      list(weak = weakref$ref(array), column = empty(np_array_column(array)))
    })
    gc()
    gc()
    expect_null(py_to_r(state$weak()))
    expect_length(state$column, 0L)
    expect_identical(dim(np_array(state$column)), c(0L, 2L))
  }

  state <- local({
    old <- np$ones(c(3L, 2L), dtype = "int8")
    replacement <- np_array_column(np$zeros(c(3L, 2L), dtype = "int8"))
    column <- vctrs::vec_assign(np_array_column(old), 1:3, replacement)
    list(weak = weakref$ref(old), column = column)
  })
  gc()
  gc()
  expect_null(py_to_r(state$weak()))
  expect_true(py_to_r(np$all(np_array(state$column) == 0L)))
})

test_that("NumPy column metadata operations do not copy array contents", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("tibble")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  tracemalloc <- import("tracemalloc", convert = FALSE)
  skip_if(py_to_r(tracemalloc$is_tracing()))
  on.exit(tracemalloc$stop(), add = TRUE)
  reverse <- py_eval("lambda x: x[:, :, ::-1]", convert = FALSE)

  # Warm the printing methods before measuring the array-size-dependent cost.
  tiny <- tibble::tibble(pixels = np_array_column(np$zeros(c(2L, 2L), dtype = "uint8")))
  invisible(capture.output(print(tiny)))

  for (layout in c("C", "F", "strided")) {
    x <- np$ones(c(64L, 1024L, 1024L), dtype = "uint8",
                  order = if (layout == "F") "F" else "C")
    if (layout == "strided")
      x <- reverse(x)
    bytes <- py_to_r(x$nbytes)

    tracemalloc$start()
    column <- np_array_column(x)
    proxy <- vctrs::vec_proxy(column)
    prototype <- vctrs::vec_ptype(column)
    df <- tibble::tibble(id = 1:64, pixels = column)
    selected <- vctrs::vec_slice(df, 1L)
    invisible(capture.output(print(df, n = 6L)))
    peak <- py_to_r(tracemalloc$get_traced_memory())[[2L]]
    tracemalloc$stop()
    expect_lt(peak, bytes / 16, label = paste(layout, "metadata allocation"))
    expect_length(prototype, 0L)

    # Gathering a selected row may allocate that row, but not the full source.
    tracemalloc$start()
    result <- np_array(selected$pixels)
    peak <- py_to_r(tracemalloc$get_traced_memory())[[2L]]
    tracemalloc$stop()
    expect_lt(peak, bytes / 16, label = paste(layout, "one-row allocation"))
    expect_identical(dim(result), c(1L, 1024L, 1024L))
    expect_true(py_to_r(np$all(result == 1L)))
  }
})

test_that("NumPy conversion avoids redundant buffers after metadata materialization", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  tracemalloc <- import("tracemalloc", convert = FALSE)
  skip_if(py_to_r(tracemalloc$is_tracing()))
  on.exit(tracemalloc$stop(), add = TRUE)

  x <- np$ones(c(16L, 512L, 512L), dtype = "uint8", order = "F")
  column <- np_array_column(x)
  invisible(vctrs::vec_chop(column, list(1:8, 9:16)))
  bytes <- py_to_r(x$nbytes)
  tracemalloc$start()
  result <- r_to_py(column)
  peak <- py_to_r(tracemalloc$get_traced_memory())[[2L]]
  tracemalloc$stop()
  expect_identical(py_id(result), py_id(x))
  expect_lt(peak, bytes / 2)

  y <- np$zeros(c(16L, 512L, 512L), dtype = "uint8", order = "F")
  bound <- vctrs::vec_c(column, np_array_column(y))
  tracemalloc$start()
  result <- r_to_py(bound)
  peak <- py_to_r(tracemalloc$get_traced_memory())[[2L]]
  tracemalloc$stop()
  # Allocate the combined result, without an extra copy of either full input.
  expect_lt(peak, bytes * 2.5)
  expect_true(py_to_r(np$array_equal(result, np$concatenate(list(x, y)))))
})

test_that("binding and assigning NumPy columns avoid Python objects per row", {
  skip_if_no_numpy()
  skip_if(getRversion() < "4.3.0")
  skip_if_not_installed("tibble")
  skip_if_not_installed("dplyr")
  skip_if_not_installed("vctrs", "0.7.0")
  np <- import("numpy", convert = FALSE)
  tracemalloc <- import("tracemalloc", convert = FALSE)
  skip_if(py_to_r(tracemalloc$is_tracing()))
  on.exit(tracemalloc$stop(), add = TRUE)

  x <- np$arange(20000L, dtype = "int32")
  column <- np_array_column(x)
  df <- tibble::tibble(pixels = column)
  tracemalloc$start()
  bound <- dplyr::bind_rows(df, df)
  assigned <- vctrs::vec_assign(column, 1L, column[20000L])
  groups <- vctrs::vec_chop(column, list(1:10000, 10001:20000))
  peak <- py_to_r(tracemalloc$get_traced_memory())[[2L]]
  tracemalloc$stop()
  expect_lt(peak, 1024^2)
  expect_length(bound$pixels, 40000L)
  expect_identical(py_to_r(np_array(assigned)[0L]), 19999L)
  expect_identical(vapply(groups, length, integer(1)), c(10000L, 10000L))
  expect_true(py_to_r(np$array_equal(np_array(bound$pixels), np$concatenate(list(x, x)))))
})
