context("helper handlers")

test_that("completion help formats empty and nested sequence defaults", {
  skip_if_no_python()
  scope <- py_run_string("
class Defaults:
    def empty(values=[], pair=(), members=set()):
        pass

    def nested(values=[(1, True), ['two', None]], members={3}):
        pass
", local = TRUE, convert = FALSE)

  expect_identical(
    py_help_handler("completion", "empty", scope$Defaults)$signature,
    "empty(values = c(), pair = list(), members = list())"
  )
  expect_identical(
    py_help_handler("completion", "nested", scope$Defaults)$signature,
    'nested(values = c(list(1L, TRUE), c("two", NULL)), members = list(3L))'
  )
})

test_that("completion help closes large sequence defaults", {
  skip_if_no_python()
  scope <- py_run_string("
class Defaults:
    def large(values=list(range(258))):
        pass
", local = TRUE, convert = FALSE)

  expected <- paste0("large(values = c(", paste0(0:257, "L", collapse = ", "), "))")
  expect_identical(
    py_help_handler("completion", "large", scope$Defaults)$signature,
    expected
  )
})

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
  expect_setequal(names(doctree$ids), c("parameters", "returns", "section1"))

  arg_descriptions <- arg_descriptions_from_doc_sphinx(docs)
  expect_setequal(names(arg_descriptions), c("a", "type"))
  expect_match(arg_descriptions[["a"]], "Description for a")
  expect_match(arg_descriptions[["type"]], "Type of algorithm")

  result <- help_completion_handler_sphinx(docs)
  expect_match(result$description, "Initialize the model")
  expect_match(result$returns, "values: array-like")
})

test_that("Documentations in Google style can be extracted correctly for help handlers", {
  skip_if_no_docutils()

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

