test_that("multiprocessing works", {

  skip_if_no_python()

  mp <- import("multiprocessing")
  queue <- mp$Queue()

  for (i in 1:3) {

    expect_no_error({
      p <- mp$Process(target = queue$put,
                      args = tuple(i))
      p$start()
      p$join()
    })

    expect_equal(p$exitcode, 0L)

  }

  expect_equal(queue$get(), 1L)
  expect_equal(queue$get(), 2L)
  expect_equal(queue$get(), 3L)


})
