context("datetime")

test_that("R dates can be converted to / from Python datetimes", {
  skip_if_no_python()

  before <- list(Sys.Date(), as.Date("2019-04-06"))
  after  <- py_to_r(r_to_py(before))

  expect_equal(before, after)
})

test_that("R times can be converted to / from Python datetimes", {
  skip_if_no_numpy()

  before <- Sys.time()
  attr(before, "tzone") <- "UTC"
  after <- py_to_r(r_to_py(before))

  expect_equal(as.numeric(before), as.numeric(after))
})

test_that("lists of times are converted", {
  skip_if_no_python()

  dates <- replicate(3, Sys.Date(), simplify = FALSE)
  expect_equal(
    py_to_r(r_to_py(dates)),
    dates
  )

})

test_that("R times are converted to NumPy datetime64", {
  skip_if_no_numpy()

  np <- import("numpy", convert = TRUE)

  before <- rep(Sys.time(), 3)
  converted <- r_to_py(before)
  expect_true(np$issubdtype(converted$dtype, np$datetime64))

  after <- py_to_r(converted)
  expect_equal(
    as.numeric(as.POSIXct(before)),
    as.numeric(as.POSIXct(after))
  )

})

test_that("R datetimes can be passed to Python functions", {
  skip_if_no_python()
  py_run_string("def identity(x): return x")
  main <- import_main()
  date <- Sys.Date()
  expect_equal(date, main$identity(Sys.Date()))
})

test_that("timezone information is not lost during conversion", {

  skip_if_no_python()
  skip_if(py_version() < 3)

  datetime <- import("datetime", convert = FALSE)
  if (!py_has_attr(datetime, "timezone"))
    skip("datetime.timezone is not available")

  pdt <- datetime$datetime(
    year   = 2020L,
    month  = 8L,
    day    = 24L,
    hour   = 3L,
    minute = 4L,
    second = 5L,
    tzinfo = datetime$timezone$utc
  )

  rdt <- py_to_r(pdt)

  tzone <- attr(rdt, "tzone", exact = TRUE)
  expect_identical(tzone, "UTC")

  expect_identical(
    format(rdt, "%Y-%m-%dT%H:%M:%S%z"),
    "2020-08-24T03:04:05+0000")
  # py_to_r(pdt$isoformat(timespec="seconds")))

  if(py_version() >= "3.9") {
    zoneinfo <- import("zoneinfo")
    pdt <- datetime$datetime$now(zoneinfo$ZoneInfo("America/New_York"))
    rdt <- py_to_r(pdt)
    expect_identical(attr(rdt, "tzone", TRUE), "America/New_York")
    expect_identical(format(rdt, "%Y-%m-%dT%H:%M:%S%z"),
                     py_to_r(pdt$strftime("%Y-%m-%dT%H:%M:%S%z")))
  }

})
