
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

  if (file.exists("/usr/bin/python3")) {

    info <- python_info("/usr/bin/python3")

    expected <- list(
      python = "/usr/bin/python3",
      type = "system",
      root = "/usr/bin"
    )

    expect_equal(info, expected)

  }
})
