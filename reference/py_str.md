# String representation of a python object.

This is equivalent to calling `str(object)` or `repr(object)` in Python.

## Usage

``` r
py_repr(object)

py_str(object, ...)
```

## Arguments

- object:

  Python object

- ...:

  Unused

## Value

Character vector

## Details

In Python, calling [`print()`](https://rdrr.io/r/base/print.html)
invokes the builtin [`str()`](https://rdrr.io/r/utils/str.html), while
auto-printing an object at the REPL invokes the builtin `repr()`.

In R, the default print method for python objects invokes `py_repr()`,
and the default [`format()`](https://rdrr.io/r/base/format.html) and
[`as.character()`](https://rdrr.io/r/base/character.html) methods invoke
`py_str()`.

For historical reasons, `py_str()` is also an R S3 method that allows R
authors to customize the the string representation of a Python object
from R. New code is recommended to provide a
[`format()`](https://rdrr.io/r/base/format.html) and/or
[`print()`](https://rdrr.io/r/base/print.html) S3 R method for python
objects instead.

The default implementation will call `PyObject_Str` on the object.

## See also

[`as.character.python.builtin.str()`](https://rstudio.github.io/reticulate/reference/as.character.python.builtin.str.md)
[`as.character.python.builtin.bytes()`](https://rstudio.github.io/reticulate/reference/as.character.python.builtin.bytes.md)
for handling `Error : Embedded NUL in string.` if the Python string
contains an embedded `NUL`.
