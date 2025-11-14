# `nameOfClass()` for Python objects

This generic enables passing a `python.builtin.type` object as the 2nd
argument to [`base::inherits()`](https://rdrr.io/r/base/class.html).

## Usage

``` r
# S3 method for class 'python.builtin.type'
nameOfClass(x)
```

## Arguments

- x:

  A Python class

## Value

A scalar string matching the S3 class of objects constructed from the
type.

## Examples

``` r
if (FALSE) { # \dontrun{
  numpy <- import("numpy")
  x <- r_to_py(array(1:3))
  inherits(x, numpy$ndarray)
} # }
```
