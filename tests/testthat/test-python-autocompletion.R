context("autocompletion")

test_that("autocompletion of dictionary keys works", {
  skip_if_no_python()
  
  py_run_string("alpha = {'b1': 1, 'b2': 2, 'b3': 3}")
  completions <- py_completer("alpha['b")
  expect_equal(c(completions), c("b1", "b2", "b3"))
  expect_equal(attr(completions, "token"), "b")
  
})

test_that("autocompletion of modules works", {
  skip_if_no_python()
  
  completions <- py_completer("import s")
  expect_true("sys" %in% completions)
  expect_equal(attr(completions, "token"), "s")
})

