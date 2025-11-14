# Create Python dictionary

Create a Python dictionary object, including a dictionary whose keys are
other Python objects rather than character vectors.

## Usage

``` r
dict(..., convert = FALSE)

py_dict(keys, values, convert = FALSE)
```

## Arguments

- ...:

  Name/value pairs for dictionary (or a single named list to be
  converted to a dictionary).

- convert:

  `TRUE` to automatically convert Python objects to their R equivalent.
  If you pass `FALSE` you can do manual conversion using the
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  function.

- keys:

  Keys to dictionary (can be Python objects)

- values:

  Values for dictionary

## Value

A Python dictionary

## Note

The returned dictionary will not automatically convert its elements from
Python to R. You can do manual conversion with the
[`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
function or pass `convert = TRUE` to request automatic conversion.
