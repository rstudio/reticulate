# Wrap an R function in a Python function with the same signature.

This function could wrap an R function in a Python function with the
same signature. Note that the signature of the R function must not
contain esoteric Python-incompatible constructs.

## Usage

``` r
py_func(f)
```

## Arguments

- f:

  An R function

## Value

A Python function that calls the R function `f` with the same signature.
