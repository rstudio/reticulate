context("dill")

source("helper-utils.R")

test_that("Interpreter sessions can be saved and loaded with dill", {
  skip_if_no_python()
  skip_if_not_installed("callr")
  
  session_one_vars <- callr::r(
    function() {
      module_load <- tryCatch(
        dill <- reticulate::import("dill"),
        error = function(c) {
          py_error <- reticulate::py_last_error()
          if(py_error$type == "ImportError" && py_error$value == "No module named dill") {
            "No dill"
          }})
      if (module_load == "No dill") return(module_load)
      main <- reticulate::py_run_string("x = 1")
      reticulate::py_run_string("y = x + 1")
      dill$dump_session(filename = "x.dill", byref = TRUE)
      c(main$x, main$y)
    })
  if (session_one_vars[1] == "No dill")
    skip("The dill Python module is not installed")
  
  session_two_vars <- callr::r(
    function() {
      dill <- reticulate::import("dill")
      dill$load_session(filename = "x.dill")
      main <- reticulate::py_run_string("pass")
      c(main$x, main$y)
    })
  on.exit(unlink("x.dill"), add = TRUE)
  expect_equal(session_one_vars, session_two_vars)
})

