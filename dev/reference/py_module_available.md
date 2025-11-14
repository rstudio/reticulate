# Check if a Python module is available on this system.

Note that this function will also attempt to initialize Python before
checking if the requested module is available.

## Usage

``` r
py_module_available(module)
```

## Arguments

- module:

  The name of the module.

## Value

`TRUE` if the module is available and can be loaded; `FALSE` otherwise.
