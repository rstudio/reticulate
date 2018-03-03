context("functions")

source("utils.R")

test_that("Sphinx style documentations can be extracted correctly for help handlers", {
  skip_if_no_docutils()

  docs <- "
Initialize the model.

Parameters
----------
a : int, optional
  Description for a.
  Defaults to 3.
type : str, optional
  Type of algorithm (default: 'linear')
    'linear'        - linear model
    'nonlinear'     - nonlinear model

Returns
-------
values : array-like

Section1
-----------
Just a placeholder here.
"

  doctree <- sphinx_doctree_from_doc(docs)
  expect_equal(names(doctree$ids), c("parameters", "returns", "section1"))

  arg_descriptions <- arg_descriptions_from_doc_sphinx(docs)
  expect_match(arg_descriptions[[1]], "Description for a")
  expect_match(arg_descriptions[[2]], "Type of algorithm")

  result <- help_completion_handler_sphinx(docs)
  expect_match(result$description, "Initialize the model")
  expect_match(result$returns, "values: array-like")
})

