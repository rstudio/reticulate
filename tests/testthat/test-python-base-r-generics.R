test_that("Ops group generics dispatch correctly for Python objects", {

  numpy <- import("numpy", convert = FALSE)

  py_obj1 <- numpy$array(c(1, 2, 3))
  py_obj2 <- numpy$array(c(4, 5, 6))

  # Test arithmetic operations
  expect_equal(py_to_r(py_obj1 + py_obj2), array(c(5, 7, 9)))
  expect_equal(py_to_r(py_obj1 - py_obj2), array(c(-3, -3, -3)))
  expect_equal(py_to_r(py_obj1 * py_obj2), array(c(4, 10, 18)))
  expect_equal(py_to_r(py_obj1 / py_obj2), array(c(1/4, 2/5, 3/6)))
  expect_equal(py_to_r(py_obj1 ^ 2), array(c(1, 4, 9)))
  expect_equal(py_to_r(py_obj1 %% 2), array(c(1, 0, 1)))
  expect_equal(py_to_r(py_obj1 %/% 2), array(c(0, 1, 1)))

  # Test logical operations
  py_bool1 <- numpy$array(c(TRUE, FALSE, TRUE))
  py_bool2 <- numpy$array(c(FALSE, TRUE, FALSE))

  expect_equal(py_to_r(py_bool1 & py_bool2), array(c(FALSE, FALSE, FALSE)))
  expect_equal(py_to_r(py_bool1 | py_bool2), array(c(TRUE, TRUE, TRUE)))
  expect_equal(py_to_r(!py_bool1), array(c(FALSE, TRUE, FALSE)))

  # Test comparison operations
  expect_equal(py_to_r(py_obj1 == py_obj2), array(c(FALSE, FALSE, FALSE)))
  expect_equal(py_to_r(py_obj1 != py_obj2), array(c(TRUE, TRUE, TRUE)))
  expect_equal(py_to_r(py_obj1 < py_obj2), array(c(TRUE, TRUE, TRUE)))
  expect_equal(py_to_r(py_obj1 <= py_obj2), array(c(TRUE, TRUE, TRUE)))
  expect_equal(py_to_r(py_obj1 >= py_obj2), array(c(FALSE, FALSE, FALSE)))
  expect_equal(py_to_r(py_obj1 > py_obj2), array(c(FALSE, FALSE, FALSE)))
})


test_that("Ops group generics dispatch correctly when only one argument is a Python object", {
  numpy <- import("numpy", convert = FALSE)
  py_obj <- numpy$array(c(1, 2, 3))
  r_obj <- array(c(4, 5, 6))

  # Test arithmetic operations
  expect_equal(py_to_r(py_obj + r_obj), array(c(5, 7, 9)))
  expect_equal(py_to_r(r_obj + py_obj), array(c(5, 7, 9)))

  expect_equal(py_to_r(py_obj - r_obj), array(c(-3, -3, -3)))
  expect_equal(py_to_r(r_obj - py_obj), array(c(3, 3, 3)))

  expect_equal(py_to_r(py_obj * r_obj), array(c(4, 10, 18)))
  expect_equal(py_to_r(r_obj * py_obj), array(c(4, 10, 18)))

  expect_equal(py_to_r(py_obj / r_obj), array(c(1/4, 2/5, 3/6)))
  expect_equal(py_to_r(r_obj / py_obj), array(c(4, 5/2, 2)))

  expect_equal(py_to_r(py_obj ^ 2), array(c(1, 4, 9)))
  expect_equal(py_to_r(r_obj ^ py_obj), array(c(4, 25, 216)))

  expect_equal(py_to_r(py_obj %% 2), array(c(1, 0, 1)))
  expect_equal(py_to_r(r_obj %% py_obj), array(c(0, 1, 0)))

  expect_equal(py_to_r(py_obj %/% 2), array(c(0, 1, 1)))
  expect_equal(py_to_r(r_obj %/% py_obj), array(c(4, 2, 2)))

  # Test logical operations
  py_bool <- numpy$array(array(c(TRUE, FALSE, TRUE)))
  r_bool <- array(c(FALSE, TRUE, FALSE))

  expect_equal(py_to_r(py_bool & r_bool), array(c(FALSE, FALSE, FALSE)))
  expect_equal(py_to_r(r_bool & py_bool), array(c(FALSE, FALSE, FALSE)))

  expect_equal(py_to_r(py_bool | r_bool), array(c(TRUE, TRUE, TRUE)))
  expect_equal(py_to_r(r_bool | py_bool), array(c(TRUE, TRUE, TRUE)))

  expect_equal(py_to_r(!py_bool), array(c(FALSE, TRUE, FALSE)))

  # Test comparison operations
  expect_equal(py_to_r(py_obj == r_obj), array(c(FALSE, FALSE, FALSE)))
  expect_equal(py_to_r(r_obj == py_obj), array(c(FALSE, FALSE, FALSE)))

  expect_equal(py_to_r(py_obj != r_obj), array(c(TRUE, TRUE, TRUE)))
  expect_equal(py_to_r(r_obj != py_obj), array(c(TRUE, TRUE, TRUE)))

  expect_equal(py_to_r(py_obj < r_obj), array(c(TRUE, TRUE, TRUE)))
  expect_equal(py_to_r(r_obj < py_obj), array(c(FALSE, FALSE, FALSE)))

  expect_equal(py_to_r(py_obj <= r_obj), array(c(TRUE, TRUE, TRUE)))
  expect_equal(py_to_r(r_obj <= py_obj), array(c(FALSE, FALSE, FALSE)))

  expect_equal(py_to_r(py_obj >= r_obj), array(c(FALSE, FALSE, FALSE)))
  expect_equal(py_to_r(r_obj >= py_obj), array(c(TRUE, TRUE, TRUE)))

  expect_equal(py_to_r(py_obj > r_obj), array(c(FALSE, FALSE, FALSE)))
  expect_equal(py_to_r(r_obj > py_obj), array(c(TRUE, TRUE, TRUE)))
})

if (getRversion() >= "4.3.0")
test_that("matrixOps group generics dispatch", {

  r_obj1 <- array(1:9, c(3, 3))
  r_obj2 <- t(r_obj1) + 10

  py_obj1 <- r_to_py(r_obj1)
  py_obj2 <- r_to_py(r_obj2)

  expect_equal(py_to_r(py_obj1 %*% py_obj2), r_obj1 %*% r_obj2)
  expect_equal(py_to_r(py_obj2 %*% py_obj1), r_obj2 %*% r_obj1)

  expect_equal(py_to_r(py_obj1 %*% r_obj2), r_obj1 %*% r_obj2)
  expect_equal(py_to_r(py_obj2 %*% r_obj1), r_obj2 %*% r_obj1)

  expect_equal(py_to_r(r_obj1 %*% py_obj2), r_obj1 %*% r_obj2)
  expect_equal(py_to_r(r_obj2 %*% py_obj1), r_obj2 %*% r_obj1)

})


test_that("[ can infer slices, multiple args", {

  x <- r_to_py(array(1:64, c(4, 4, 4)))
  py$x <- x

  expect_identical(py_eval("x[0]"), py_to_r(x[0]))
  expect_identical(py_eval("x[:, 0]"), py_to_r(x[, 0]))
  expect_identical(py_eval("x[:, :, 0]"), py_to_r(x[, , 0]))

  expect_identical(py_eval("x[:2]"), py_to_r(x[`:2`]))
  expect_identical(py_eval("x[:2]"), py_to_r(x[NULL:2]))
  expect_identical(py_eval("x[:2]"), py_to_r(x[NA:2]))

  expect_identical(py_eval("x[1:2]"), py_to_r(x[1:2]))
  expect_identical(py_eval("x[1:2]"), py_to_r(x[`1:2`]))

  expect_identical(py_eval("x[2:]"), py_to_r(x[2:NA]))
  expect_identical(py_eval("x[2:]"), py_to_r(x[`2:`]))
  expect_identical(py_eval("x[2:]"), py_to_r(x[2:NULL]))

  expect_identical(py_eval("x[1:3:2]"), py_to_r(x[1:3:2]))
  expect_identical(py_eval("x[1:3:2]"), py_to_r(x[`1:3:2`]))

  expect_identical(py_eval("x[::2]"), py_to_r(x[`::2`]))
  expect_identical(py_eval("x[::2]"), py_to_r(x[NULL:NULL:2]))
  expect_identical(py_eval("x[::2]"), py_to_r(x[NA:NA:2]))

  expect_identical(py_eval("x[:, :2]"), py_to_r(x[, `:2`]))
  expect_identical(py_eval("x[:, :2]"), py_to_r(x[, NULL:2]))
  expect_identical(py_eval("x[:, :2]"), py_to_r(x[, NA:2]))

  expect_identical(py_eval("x[:, ::2]"), py_to_r(x[, `::2`]))
  expect_identical(py_eval("x[:, ::2]"), py_to_r(x[, NULL:NULL:2]))
  expect_identical(py_eval("x[:, ::2]"), py_to_r(x[, NA:NA:2]))

  expect_identical(py_eval("x[:, :2, :]"), py_to_r(x[, `:2`]))
  expect_identical(py_eval("x[:, :2, :]"), py_to_r(x[, NULL:2]))
  expect_identical(py_eval("x[:, :2, :]"), py_to_r(x[, NA:2]))

  expect_identical(py_eval("x[:, ::2, :]"), py_to_r(x[, `::2`, ]))
  expect_identical(py_eval("x[:, ::2, :]"), py_to_r(x[, NULL:NULL:2, ]))
  expect_identical(py_eval("x[:, ::2, :]"), py_to_r(x[, NA:NA:2, ]))

  # test the test is actually comparing R arrays
  py$x <- x <- np_array(x, dtype = "float64") # https://github.com/rstudio/reticulate/issues/1473
  expect_identical(py_to_r(x[, , 0]), array(as.double(1:16), c(4, 4)))
  expect_identical(py_eval("x[:, :, 0]"), array(as.double(1:16), c(4, 4)))

  # copy `x` to make it writeable
  py_run_string("import numpy as np; x = np.array(x)")
  x <- np_array(x)

  py_run_string("x[:, 2, :] = 99")
  x[, 2, ] <- 99L
  expect_identical(py_eval("x"), py_to_r(x))

})


test_that("[ passes through python objects", {

  skip_if_no_numpy("numpy")

  np <- import("numpy", convert = FALSE)

  x <- np$arange(10L)
  ir <- 3L
  ip <- np$array(3L)

  expect_equal(py_to_r(x[ir]), 3L)
  expect_equal(py_to_r(x[ip]), 3L)

  expect_equal(py_to_r(x[ir:NA]), array(3:9))
  expect_equal(py_to_r(x[ip:NA]), array(3:9))
  expect_equal(py_to_r(x[NA:ir]), array(0:2))
  expect_equal(py_to_r(x[NA:ip]), array(0:2))

})
