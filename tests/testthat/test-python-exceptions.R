context("exceptions")


test_that("py_last_error() returns R strings", {
  skip_if_no_python()

  tryCatch(py_eval("range(3)[3]"), error = identity)

  er <- py_last_error()
  expect_identical(er$type, "IndexError")
  expect_type(er$value, "character")
  expect_type(er$traceback, "character")
  expect_type(er$message, "character")

})


test_that("py_last_error() returns the R error condition object", {
    skip_if_no_python()
    f1 <- py_func(function() {
      stop("Hello")
    })
    f2 <- py_func(function() f1())
    f3 <- py_func(function() f2())



    caught_e <- tryCatch(f1(), error = function(e) e)
    e <- py_last_error()
    rc <- e$r_condition
    expect_s3_class(rc, c("error", "condition"))
    expect_match(rc$message, "Hello")
    expect_type(rc$call, "language")

    caught_e <- tryCatch(f3(), error = function(e) e)
    e <- py_last_error()
    rc <- e$r_condition
    expect_s3_class(rc, c("error", "condition"))
    expect_match(rc$message, "Hello")
    skip("R conditions are lost when Exceptions are rethrown")

    ## TODO: R conditions are lost when Exceptions are rethrown
    ## expect_type(rc$call, "language") # works w/ f1(), but not f3()

    ## TODO: the r condition message accumulates a final \n for
    ## each py -> r stack transition, investigate what's
    ## adding it and remove

    ## TODO: `rc$call` should really be propagated through to the error
    ## condition that is then raised from the Rcpp error handler. right
    ## now, caught_e$call is NULL, should be identical to rc$call
})
