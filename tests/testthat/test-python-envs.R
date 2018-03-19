context("envs")

source("helper-utils.R")

test_that("conda utility functions work as expected", {
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
  
})

test_that("virtualenv utility functions work as expected", {
  skip_if_no_test_environments()
  
  virtualenv_remove('reticulate-testthat', confirm = FALSE)
  
  virtualenv_create('reticulate-testthat')
  virtualenv_remove('reticulate-testthat', confirm = FALSE)
  
  virtualenv_install('reticulate-testthat', 'Pillow')
  virtualenv_install('reticulate-testthat', 'Pillow', ignore_installed = TRUE)
  
  expect_true('reticulate-testthat' %in% virtualenv_list())
  
  virtualenv_remove('reticulate-testthat', confirm = FALSE)
  
  expect_false('reticulate-testthat' %in% virtualenv_list())
  
})

