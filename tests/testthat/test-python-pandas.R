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