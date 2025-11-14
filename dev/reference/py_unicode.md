# Convert to Python Unicode Object

Convert to Python Unicode Object

## Usage

``` r
py_unicode(str)
```

## Arguments

- str:

  Single element character vector to convert

## Details

By default R character vectors are converted to Python strings. In
Python 3 these values are unicode objects however in Python 2 they are
8-bit string objects. This function enables you to obtain a Python
unicode object from an R character vector when running under Python 2
(under Python 3 a standard Python string object is returned).
