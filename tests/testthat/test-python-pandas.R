context("pandas")

test_that("Simple Pandas data frames can be roundtripped", {
  skip_if_no_pandas()
  
  pd <- import("pandas")
  
  before <- iris
  after  <- py_to_r(r_to_py(before))
  mapply(function(lhs, rhs) {
    expect_equal(lhs, rhs)
  }, before, after)
  
})

test_that("Ordered factors are preserved", {
  skip_if_no_pandas()
  
  pd <- import("pandas")
  
  set.seed(123)
  before <- data.frame(x = ordered(letters, levels = sample(letters)))
  after <- py_to_r(r_to_py(before))
  expect_equal(before, after)
  
})

test_that("Generic methods for pandas objects produce correct results", {
  skip_if_no_pandas()

  df <- data.frame(x = c(1, 3), y = c(4, 4), z = c(5, 5))
  pdf <- r_to_py(df)

  expect_equal(length(pdf), length(df))
  expect_equal(length(pdf$x), length(df$x))
  
  expect_equal(dim(pdf), dim(df))
  expect_equal(dim(pdf$x), dim(df$x))

  expect_equal(dim(summary(pdf)), c(8, 3))
  expect_equal(length(summary(pdf$x)), 8)
})

test_that("Timestamped arrays in Pandas DataFrames can be roundtripped", {
  skip_if_no_pandas()
  
  pd <- import("pandas", convert = FALSE)
  np <- import("numpy", convert = FALSE)
  
  data <- list(
    'A' = 1.,
    'B' = pd$Timestamp('20130102'),
    'C' = pd$Series(1:4, dtype = 'float32'),
    'D' = np$array(rep(3L, 4), dtype = 'int32'),
    'E' = pd$Categorical(c("test", "train", "test", "train")),
    'F' = 'foo'
  )
  
  before <- pd$DataFrame(data)
  
  converted <- py_to_r(before)
  
  after <- r_to_py(converted)
  
  expect_equal(py_to_r(before$to_csv()), py_to_r(after$to_csv()))
  
})

test_that("data.frames with length-one factor columns can be converted", {
  skip_if_no_pandas()
  
  pd <- import("pandas", convert = FALSE)
  np <- import("numpy", convert = FALSE)
  
  before <- data.frame(x = "hello")
  converted <- r_to_py(before)
  after <- py_to_r(converted)
  
  expect_identical(before, after)
  
})
