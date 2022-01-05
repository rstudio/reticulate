
test_that("python_info() accepts system python", {

  if (file.exists("/usr/bin/python")) {

    info <- python_info("/usr/bin/python")

    expected <- list(
      python = "/usr/bin/python",
      type = "system",
      root = "/usr/bin"
    )

    expect_equal(info, expected)

  }
})
