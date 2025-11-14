# Run Python code

Execute code within the scope of the `__main__` Python module.

## Usage

``` r
py_run_string(code, local = FALSE, convert = TRUE)

py_run_file(file, local = FALSE, convert = TRUE, prepend_path = TRUE)
```

## Arguments

- code:

  The Python code to be executed.

- local:

  Boolean; should Python objects be created as part of a local / private
  dictionary? If `FALSE`, objects will be created within the scope of
  the Python main module.

- convert:

  Boolean; should Python objects be automatically converted to their R
  equivalent? If set to `FALSE`, you can still manually convert Python
  objects to R via the
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  function.

- file:

  The Python script to be executed.

- prepend_path:

  Boolean; should the script directory be added to the Python module
  search path? The default, `TRUE`, matches the behavior of
  `python <path/to/script.py>` at the command line.

## Value

A Python dictionary of objects. When `local` is `FALSE`, this dictionary
captures the state of the Python main module after running the provided
code. Otherwise, only the variables defined and used are captured.
