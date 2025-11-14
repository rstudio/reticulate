# Python Truthiness

Equivalent to `bool(x)` in Python, or `not not x`.

## Usage

``` r
py_bool(x)
```

## Arguments

- x, :

  A python object.

## Value

An R scalar logical: `TRUE` or `FALSE`. If `x` is a null pointer or
Python is not initialized, `FALSE` is returned.

## Details

If the Python object defines a `__bool__` method, then that is invoked.
Otherwise, if the object defines a `__len__` method, then `TRUE` is
returned if the length is nonzero. If neither `__len__` nor `__bool__`
are defined, then the Python object is considered `TRUE`.
