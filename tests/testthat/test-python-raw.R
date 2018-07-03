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

