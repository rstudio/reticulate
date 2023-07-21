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
  expect_equal(before, after, check.attributes = FALSE)

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

  # TODO: this test fails on Windows because the int32 array gets
  # converted to an R numeric vector rather than an integer vector
  skip_on_os("windows")

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

  expect_equal(before, after, check.attributes = FALSE)

})

test_that("py_to_r preserves a Series index as names", {
  skip_if_no_pandas()

  pd <- import("pandas", convert = FALSE)
  np <- import("numpy", convert = FALSE)

  index <- c("a", "b", "c", "d", "e")
  values <- rnorm(5)

  s <- pd$Series(values, index = as.list(index))
  s$name <- "hi"

  r <- py_to_r(s)
  expect_equal(as.numeric(r), values)
  expect_identical(names(r), index)

})

test_that("complex names are handled", {
  skip_if_no_pandas()

  pd <- import("pandas", convert = FALSE)

  d <- dict(col1 = list(1,2))

  d[tuple("col1", "col2")] <- list(4, 5)

  p <- pd$DataFrame(data = d)
  r <- py_to_r(p)
  expect_equal(names(r), c("col1", "(col1, col2)"))

})

test_that("single-row data.frames with rownames can be converted", {
  skip_if_no_pandas()

  before <- data.frame(A = 1, row.names = "ID01")
  after <- py_to_r(r_to_py(before))
  expect_equal(c(before), c(after))

})

test_that("Time zones are respected if available", {
  skip_if_no_pandas()

  pd <- import("pandas", convert = FALSE)

  before <- pd$DataFrame(list('TZ' = pd$Series(
    c(
      pd$Timestamp('20130102003020', tz = 'US/Pacific'),
      pd$Timestamp('20130102003020', tz = 'CET'),
      pd$Timestamp('20130102003020', tz = 'UTC'),
      pd$Timestamp('20130102003020', tz = 'Hongkong')
    )
  )))

  converted <- py_to_r(before)
  after <- r_to_py(converted)

  # check if both are the same in *local* timezone
  expect_equal(py_to_r(before), py_to_r(after))

})

test_that("NaT is converted to NA", {
  skip_if_no_pandas()

  pd <- import("pandas", convert = FALSE)
  np <- import("numpy")

  before <- pd$DataFrame(pd$Series(
    c(
      pd$Timestamp(NULL),
      pd$Timestamp(np$nan)
    )
  ))

  converted <- py_to_r(before)
  after <- r_to_py(converted)

  expect_equal(py_to_r(before), py_to_r(after))


})

test_that("Large ints are handled correctly", {
  skip_if_no_pandas()
  require(bit64)

  options(reticulate.long_as_bit64=TRUE)
  options(reticulate.ulong_as_bit64=TRUE)
  A <- data.frame(val=c(as.integer64("1567447722123456785"), as.integer64("1567447722123456786")))
  expect_equal(A, py_to_r(r_to_py(A)))

  options(reticulate.long_as_bit64=F)
  options(reticulate.ulong_as_bit64=F)
  A <- data.frame(val=c(as.integer64("1567447722123456785"), as.integer64("1567447722123456786")))
  expect_equal(A, py_to_r(r_to_py(A)))

})

test_that("pandas NAs are converted to R NAs", {
  skip_if_no_pandas()

  code <- "
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, pd.NA]})
"

  locals <- py_run_string(code, local = TRUE, convert = TRUE)

  df <- locals$df
  expect_true(is.na(df$b[[3]]))

  pd <- import("pandas", convert = FALSE)
  pdNA <- py_to_r(py_get_attr(pd, "NA"))
  expect_true(is.na(pdNA))

})

test_that("categorical NAs are handled", {
  skip_if_no_pandas()

  df <- data.frame(x = factor("a", NA))
  pdf <- r_to_py(df)
  rdf <- py_to_r(pdf)
  attr(rdf, "pandas.index") <- NULL
  expect_equal(df, rdf)

})



test_that("ordered categoricals are handled correctly, #1234", {
  skip_if_no_pandas()

  p_df <- py_run_string(
'import pandas as pd

# Create Dataframe with Unordered & Ordered Factors
df = pd.DataFrame({"FCT": pd.Categorical(["No", "Yes"]),
                   "ORD": pd.Categorical(["No", "Yes"], ordered=True)})
', local = TRUE)$df

  r_df <- data.frame("FCT" = factor(c("No", "Yes")),
                     "ORD" = factor(c("No", "Yes"), ordered = TRUE))

  attr(p_df, "pandas.index") <- NULL

  expect_identical(p_df, r_df)


})
