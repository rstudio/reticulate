context("helper handlers")

test_that("Documentations in Sphinx style can be extracted correctly for help handlers", {
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
  expect_equal(names(arg_descriptions), c("a", "type"))
  expect_match(arg_descriptions[["a"]], "Description for a")
  expect_match(arg_descriptions[["type"]], "Type of algorithm")

  result <- help_completion_handler_sphinx(docs)
  expect_match(result$description, "Initialize the model")
  expect_match(result$returns, "values: array-like")
})

test_that("Documentations in Google style can be extracted correctly for help handlers", {

  docs <- "
Initialize the model.
  
Args:
  a: Description for a.
    Defaults to 3.
  type: Type of algorithm (default: 'linear')
    'linear'        - linear model
    'nonlinear'     - nonlinear model
  
Returns:
  array-like values
  
Section1:
  Just a placeholder here.
"

  sections <- sections_from_doc(docs)
  expect_equal(names(sections), c("Args", "Returns", "Section1"))

  arg_descriptions <- arg_descriptions_from_doc_default(c("a", "type"), docs)
  expect_match(arg_descriptions[["a"]], "Description for a")
  expect_match(arg_descriptions[["type"]], "Type of algorithm")
  
  result <- help_completion_handler_default(docs)
  expect_match(result$description, "Initialize the model")
  expect_match(result$returns, "array-like values")
})

test_that("Parameter help can be extracted from Python objects", {
  skip_if_no_docutils()
  
  py_run_string('
def foo(alpha, omega):
  """
Parameters
----------

alpha: int, optional
  The first argument.

omega: int, optional
  The last argument.
"""
  pass
')
  
  output <- help_completion_parameter_handler.python.builtin.object(main$foo)
  expect_equal(output$args, c("alpha", "omega"))
  expect_match(output$arg_descriptions[[1]], "The first argument.")
  expect_match(output$arg_descriptions[[2]], "The last argument.")
})