context("Python Engine")

test_that("The Python engine discovers variables that are used but not defined", {
  
  ast <- import("ast", convert = TRUE)
  code <- readLines("resources/python-ast.py")
  node <- ast$parse(paste(code, collapse = "\n"))
  body <- node$body
  options(error = recover)
  
  context <- new.env(parent = emptyenv())
  eng_python_analyze(code, context = context)
})