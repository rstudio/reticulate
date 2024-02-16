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
  expect_equal(names(r)[1], "col1")
  # pandas 2.2 removed index.format(), and users must pass custom formatters now,
  # we default to using __str__ for formatting, which given a tuple, falls back
  # to __repr__ (which prints strings with quotes).
  expect_in(names(r)[2], c("(col1, col2)", "('col1', 'col2')"))

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

test_that("can cast from pandas nullable types", {
  skip_if_no_pandas()
  pd <- import("pandas", convert = FALSE)
  data <- list(
    list(name = "Int8", type = pd$Int8Dtype(), data = list(NULL, 1L, 2L)),
    list(name = "Int16", type = pd$Int16Dtype(), data = list(NULL, 1L, 2L)),
    list(name = "Int32", type = pd$Int32Dtype(), data = list(NULL, 1L, 2L)),
    list(name = "Int64", type = pd$Int64Dtype(), data = list(NULL, 1L, 2L)),
    list(name = "UInt8", type = pd$UInt8Dtype(), data = list(NULL, 1L, 2L)),
    list(name = "UInt16", type = pd$UInt16Dtype(), data = list(NULL, 1L, 2L)),
    list(name = "UInt32", type = pd$UInt32Dtype(), data = list(NULL, 1L, 2L)),
    list(name = "UInt64", type = pd$UInt64Dtype(), data = list(NULL, 1L, 2L)),
    list(name = "boolean", type = pd$BooleanDtype(), data = list(NULL, TRUE, FALSE)),
    list(name = "string", type = pd$StringDtype(), data = list(NULL, "a", "b"))
  )

  # Float32 was added sometime after v1.1.5
  if (reticulate::py_has_attr(pd, "Float32Dtype")) {
    data <- append(
      data,
      list(
        list(name = "Float32", type = pd$Float32Dtype(), data = list(NULL, 0.5, 0.3)),
        list(name = "Float64", type = pd$Float64Dtype(), data = list(NULL, 0.5, 0.3))
      )
    )
  }

  for (el in data) {
    p_df <- pd$DataFrame(list("x" = pd$Series(el$data, dtype = el$type)))
    expect_equal(py_to_r(p_df$x$dtype$name), el$name)
    r_df <- py_to_r(p_df)

    expect_equal(
      r_df$x,
      unlist(lapply(el$data, function(x) if (is.null(x)) NA else x))
    )
  }

})

test_that("NA in string columns don't prevent simplification", {
  skip_if_no_pandas()

  pd <- import("pandas", convert = FALSE)
  np <- import("numpy", convert = FALSE)

  x <- pd$Series(list("a", pd$`NA`, NULL, np$nan))
  expect_equal(py_to_r(x$dtype$name), "object")

  r <- py_to_r(x)

  expect_equal(typeof(r), "character")
  expect_equal(as.logical(is.na(r)), c(FALSE, TRUE, TRUE, TRUE))

})

test_that("NA's are preserved in pandas columns", {
  skip_if_no_pandas()
  pd <- import("pandas")
  if (numeric_version(pd$`__version__`) < "1.5") {
    skip("Nullable data types require pandas version >= 1.5 to work fully.")
  }

  df <- data.frame(
    int = c(NA, 1:10),
    num = c(NA, rnorm(10)),
    bool = c(NA, rep(c(TRUE, FALSE), 5)),
    string = c(NA, letters[1:10])
  )

  withr::with_options(c(reticulate.pandas_use_nullable_dtypes = TRUE), {
    p_df <- r_to_py(df)
  })

  r_df <- py_to_r(p_df)

  expect_identical(r_df$num, df$num)
  expect_identical(r_df$int, df$int)
  expect_identical(r_df$bool, df$bool)
  expect_identical(r_df$string, df$string)
})

test_that("Round strip for string columns with NA's work correctly", {
  skip_if_no_pandas()
  df <- data.frame(string = c(NA, letters[1:10]))
  p <- r_to_py(df)

  expect_true(py_to_r(p$string$isna()[0]))

  r <- py_to_r(p)
  expect_true(is.na(r$string[1]))
})
