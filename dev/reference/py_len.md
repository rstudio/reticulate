# Length of Python object

Get the length of a Python object. This is equivalent to calling the
Python builtin `len()` function on the object.

## Usage

``` r
py_len(x, default = NULL)
```

## Arguments

- x:

  A Python object.

- default:

  The default length value to return, in the case that the associated
  Python object has no `__len__` method. When `NULL` (the default), an
  error is emitted instead.

## Value

The length of the object, as a numeric value.

## Details

Not all Python objects have a defined length. For objects without a
defined length, calling `py_len()` will throw an error. If you'd like to
instead infer a default length in such cases, you can set the `default`
argument to e.g. `1L`, to treat Python objects without a `__len__`
method as having length one.
