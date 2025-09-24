test_that("with() propagates errors/exceptions to Python context managers", {
  skip_if_no_python()

  py_run_string(paste(
    "class _ReticulateCaptureCtx:",
    "    def __init__(self):",
    "        self.args = None",
    "    def __enter__(self):",
    "        return self",
    "    def __exit__(self, exc_type, exc_value, exc_tb):",
    "        self.args = (exc_type, exc_value, exc_tb)",
    "        return False",
    sep = "\n"
  ))

  # check R error
  ctx <- py_eval("_ReticulateCaptureCtx()")
  expect_error(with(ctx, stop('boom')), "boom")
  args <- ctx$args
  expect_s3_class(args[[1]], "python.builtin.type")
  expect_s3_class(args[[2]], "error")

  # check python exception
  ctx <- py_eval("_ReticulateCaptureCtx()")
  expect_error(
    with(ctx, py_run_string("raise ValueError('kaboom')")),
    "kaboom"
  )
  args <- ctx$args
  expect_s3_class(args[[1]], "python.builtin.type")
  expect_s3_class(args[[2]], "python.builtin.ValueError")
  expect_s3_class(args[[3]], "python.builtin.traceback")

  py_run_string("del _ReticulateCaptureCtx")
})
