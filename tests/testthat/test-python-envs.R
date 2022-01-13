context("envs")

test_that("conda utility functions work as expected", {
  # TODO: reenable these tests
  skip_if_no_test_environments()

  binary <- conda_binary()
  expect_type(binary, "character")
  expect_length(binary, 1)

  conda_remove('reticulate-testthat')
  conda_create('reticulate-testthat')
  expect_true('reticulate-testthat' %in% conda_list()$name)

  conda_install('reticulate-testthat', 'Pillow')
  conda_remove('reticulate-testthat', 'Pillow')

  conda_remove('reticulate-testthat')
  expect_false('reticulate-testthat' %in% conda_list()$name)

  conda_create('reticulate-testthat', forge = TRUE)
  expect_true(all(grepl("conda-forge", conda_list_packages("reticulate-testthat")$channel)))
  conda_remove('reticulate-testthat')

  conda_create('reticulate-testthat', channel = c("anaconda"))
  expect_true(all(grepl("anaconda", conda_list_packages("reticulate-testthat")$channel)))
  conda_remove('reticulate-testthat')

})

test_that("virtualenv utility functions work as expected", {
  skip_if_no_test_environments()

  expect_error(
    virtualenv_remove('reticulate-testthat', confirm = FALSE),
    'Virtual environment \'reticulate-testthat\' does not exist.'
  )

  virtualenv_create('reticulate-testthat')
  virtualenv_remove('reticulate-testthat', confirm = FALSE)

  virtualenv_install('reticulate-testthat', 'Pillow')
  virtualenv_install('reticulate-testthat', 'Pillow', ignore_installed = TRUE)

  expect_true('reticulate-testthat' %in% virtualenv_list())

  virtualenv_remove('reticulate-testthat', confirm = FALSE)

  expect_false('reticulate-testthat' %in% virtualenv_list())

})
