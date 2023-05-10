context("complex numbers")

test_that("Complex scalars are converted correctly", {
  skip_if_no_python()
  z <- complex(real = stats::rnorm(1), imaginary = stats::rnorm(1))
  expect_true(test$isScalar(z))
  expect_equal(z, test$reflect(z))
})

test_that("Complex vectors are converted correctly", {
  skip_if_no_python()
  z <- complex(real = stats::rnorm(100), imaginary = stats::rnorm(100))
  expect_equal(as.list(z), test$reflect(z))
  withr::with_options(c(reticulate.simplify_lists = TRUE), {
    expect_equal(z, test$reflect(z))
  })

})


test_that("Conversion from complex matrix to numpy works correctly", {
  skip_if_no_numpy()
  m <- matrix(1i^ (-6:5), nrow = 4)
  expect_equal(m, test$reflect(m))
})

