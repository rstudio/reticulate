# Read and evaluate a Python script

Evaluate a Python script within the Python main module, then make all
public (non-module) objects within the main Python module available
within the specified R environment.

## Usage

``` r
source_python(file, envir = parent.frame(), convert = TRUE)
```

## Arguments

- file:

  The Python script to be executed.

- envir:

  The environment to assign Python objects into (for example,
  [`parent.frame()`](https://rdrr.io/r/base/sys.parent.html) or
  [`globalenv()`](https://rdrr.io/r/base/environment.html)). Specify
  `NULL` to not assign Python objects.

- convert:

  Boolean; should Python objects be automatically converted to their R
  equivalent? If set to `FALSE`, you can still manually convert Python
  objects to R via the
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  function.

## Details

To prevent assignment of objects into R, pass `NULL` for the `envir`
parameter.
