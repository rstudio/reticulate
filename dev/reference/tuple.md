# Create Python tuple

Create a Python tuple object

## Usage

``` r
tuple(..., convert = FALSE)
```

## Arguments

- ...:

  Values for tuple (or a single list to be converted to a tuple).

- convert:

  `TRUE` to automatically convert Python objects to their R equivalent.
  If you pass `FALSE` you can do manual conversion using the
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  function.

## Value

A Python tuple

## Note

The returned tuple will not automatically convert its elements from
Python to R. You can do manual conversion with the
[`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
function or pass `convert = TRUE` to request automatic conversion.
