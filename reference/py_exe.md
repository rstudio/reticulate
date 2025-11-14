# Python executable

Get the path to the Python executable that `reticulate` has been
configured to use. If Python has already been initialized, then
`reticulate` will choose the currently-active copy of Python.

## Usage

``` r
py_exe()
```

## Value

The path to the Python executable `reticulate` has been configured to
use.

## Details

This can occasionally be useful if you'd like to interact with Python
(or its modules) via a subprocess; for example you might choose to
install a package with `pip`:

    system2(py_exe(), c("-m", "pip", "install", "numpy"))

and so you can also have greater control over how these modules are
invoked.
