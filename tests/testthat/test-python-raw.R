context("raw")

raw <- as.raw(c(48:57, 0, 48:57))

test_that("Raw vectors are converted to bytearraw", {
  skip_if_no_python()
  expect_s3_class(r_to_py(raw), "python.builtin.bytearray")
})

test_that("Raw vectors with null bytes roundtrip correctly", {
  skip_if_no_python()
  ba <- r_to_py(raw)
  builtins <- import_builtins()
  expect_equal(builtins$len(ba), 21)
  expect_equal(py_to_r(ba), raw)
})

test_that("Raw vector of length zero creates length zero bytearray", {
  skip_if_no_python()
  builtins <- import_builtins()
  ba <- r_to_py(as.raw(c()))
  expect_equal(builtins$len(ba), 0)
})

test_that("bytearray of length zero creates length zero Raw", {
  skip_if_no_python()
  builtins <- import_builtins()
  expect_equal(builtins$bytearray(), raw())
})

