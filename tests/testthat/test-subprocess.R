context("subprocess module")

test_that("subprocess.Popen works", {

  subprocess <- import("subprocess")

  # needs patching on Windows in the RStudio IDE
  # https://github.com/rstudio/reticulate/issues/1448
  expect_no_error({
    subprocess$Popen(
      c("ls", "."),
      shell = FALSE,
      stderr = subprocess$PIPE,
      stdout = subprocess$PIPE
    )
  })

})
