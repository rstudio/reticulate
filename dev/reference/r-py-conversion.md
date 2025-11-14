# Convert between Python and R objects

Convert between Python and R objects

## Usage

``` r
r_to_py(x, convert = FALSE)

py_to_r(x)
```

## Arguments

- x:

  A Python object.

- convert:

  Boolean; should Python objects be automatically converted to their R
  equivalent? If set to `FALSE`, you can still manually convert Python
  objects to R via the `py_to_r()` function.

## Value

An R object, as converted from the Python object.
