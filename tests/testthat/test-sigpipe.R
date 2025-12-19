test_that("SIGPIPE does not interrupt embedded Python", {
  testthat::skip_on_os("windows")
  skip_if_no_python()

  err <- tryCatch(
    {
      py_run_string("import os; r,w=os.pipe(); os.close(r); os.write(w,b'x')")
      NULL
    },
    error = function(e) e
  )

  expect_true(inherits(err, "error"))
  expect_false(grepl(
    "ignoring SIGPIPE signal",
    conditionMessage(err),
    fixed = TRUE
  ))
  expect_true(grepl("BrokenPipeError|\\[Errno 32\\]", conditionMessage(err)))
})
