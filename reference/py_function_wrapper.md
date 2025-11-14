# Scaffold R wrappers for Python functions

Scaffold R wrappers for Python functions

## Usage

``` r
py_function_wrapper(python_function, r_prefix = NULL, r_function = NULL)

py_function_docs(python_function)
```

## Arguments

- python_function:

  Fully qualified name of Python function or class constructor (e.g.
  `tf$layers$average_pooling1d`)

- r_prefix:

  Prefix to add to generated R function name

- r_function:

  Name of R function to generate (defaults to name of Python function if
  not specified)

## Note

The generated wrapper will often require additional editing (e.g. to
convert Python list literals in the docs to R lists, to massage R
numeric values to Python integers via `as.integer` where required, etc.)
so is really intended as an starting point for an R wrapper rather than
a wrapper that can be used without modification.
