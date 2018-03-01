context("autocompletion")

test_that("autocompletion of dictionary keys works", {
  skip_if_no_python()
  
  py_run_string("alpha = {'b1': 1, 'b2': 2, 'b3': 3}")
  
  line <- "alpha['b"
  completions <- py_completer(line)
  expect_equal(c(completions), c("b1", "b2", "b3"))
  expect_equal(attr(completions, "token"), "b")
  
})

test_that("autocompletion of dictionary keys works", {
  skip_if_no_python()
  
  py_run_string("alpha = {'beta': {'gamma': {'delta': 1}}}")
  
  line <- "abc + alpha['beta']['gamma']['d"
  completions <- py_completer(line)
  expect_equal(c(completions), c("delta"))
  expect_equal(attr(completions, "token"), "d")
  
})

test_that("autocompletion of modules works", {
  skip_if_no_python()
  
  completions <- py_completer("import s")
  expect_true("sys" %in% completions)
  expect_equal(attr(completions, "token"), "s")
})

test_that("autocompletion of function arguments works", {
  skip_if_no_python()
  
  py_run_string("def foo(a1, a2, a3, b1, b2, b3): pass")
  line <- "x = 1 + foo(1, a"
  completions <- py_completer(line)
  expect_equal(c(completions), c("delta"))
  expect_equal(attr(completions, "token"), "d")
  
})