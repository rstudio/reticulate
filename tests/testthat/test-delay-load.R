context("delay-load")

test_that("imported module can be customized via delay_load", {
  
  # ensure RETICULATE_PYTHON is set for sub-process so that
  # the expected version of Python is loaded
  config <- py_config()
  withr::local_envvar(RETICULATE_PYTHON = config$python)
  
  # run in a separate process, since we want the attempted module
  # load to trigger initialization of Python and so have get_module
  # handled specially
  result <- callr::r(function() {
    
    sys <- reticulate::import(
      "invalid_module_name",
      delay_load = list(get_module = function() { "sys" })
    )
    
    is.character(sys$byteorder)
    
  })
  
  # validate expected result
  expect_true(result %in% c("little", "big"))
  
})
