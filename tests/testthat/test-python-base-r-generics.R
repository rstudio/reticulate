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


if(getRversion() >= "4.3.0")
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
