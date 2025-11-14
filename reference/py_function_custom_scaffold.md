# Custom Scaffolding of R Wrappers for Python Functions

This function can be used to generate R wrapper for a specified Python
function while allowing to inject custom code for critical parts of the
wrapper generation, such as process the any part of the docs obtained
from
[`py_function_docs()`](https://rstudio.github.io/reticulate/reference/py_function_wrapper.md)
and append additional roxygen fields. The result from execution of
`python_function` is assigned to a variable called
`python_function_result` that can also be processed by `postprocess_fn`
before writing the closing curly braces for the generated wrapper
function.

## Usage

``` r
py_function_custom_scaffold(
  python_function,
  r_function = NULL,
  additional_roxygen_fields = NULL,
  process_docs_fn = function(docs) docs,
  process_param_fn = function(param, docs) param,
  process_param_doc_fn = function(param_doc, docs) param_doc,
  postprocess_fn = function() {
 },
  file_name = NULL
)
```

## Arguments

- python_function:

  Fully qualified name of Python function or class constructor (e.g.
  `tf$layers$average_pooling1d`)

- r_function:

  Name of R function to generate (defaults to name of Python function if
  not specified)

- additional_roxygen_fields:

  A list of additional roxygen fields to write to the roxygen docs, e.g.
  `list(export = "", rdname = "generated-wrappers")`.

- process_docs_fn:

  A function to process docs obtained from
  `reticulate::py_function_docs(python_function)`.

- process_param_fn:

  A function to process each parameter needed for `python_funcion`
  before executing `python_funcion.`

- process_param_doc_fn:

  A function to process the roxygen docstring for each parameter.

- postprocess_fn:

  A function to inject any custom code in the form of a string before
  writing the closing curly braces for the generated wrapper function.

- file_name:

  The file name to write the generated wrapper function to. If `NULL`,
  the generated wrapper will only be printed out in the console.

## Examples

``` r
if (FALSE) { # \dontrun{

library(tensorflow)
library(stringr)

# Example of a `process_param_fn` to cast parameters with default values
# that contains "L" to integers
process_int_param_fn <- function(param, docs) {
  # Extract the list of parameters that have integer values as default
  int_params <- gsub(
    " = [-]?[0-9]+L",
    "",
    str_extract_all(docs$signature, "[A-z]+ = [-]?[0-9]+L")[[1]])
  # Explicitly cast parameter in the list obtained above to integer
  if (param %in% int_params) {
    param <- paste0("as.integer(", param, ")")
  }
  param
}

# Note that since the default value of parameter `k` is `1L`. It is wrapped
# by `as.integer()` to ensure it's casted to integer before sending it to `tf$nn$top_k`
# for execution. We then print out the python function result.
py_function_custom_scaffold(
  "tf$nn$top_k",
  r_function = "top_k",
  process_param_fn = process_int_param_fn,
  postprocess_fn = function() { "print(python_function_result)" })

} # }
```
