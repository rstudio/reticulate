# Check if Python is available on this system

Check if Python is available on this system

## Usage

``` r
py_available(initialize = FALSE)

py_numpy_available(initialize = FALSE)
```

## Arguments

- initialize:

  `TRUE` to attempt to initialize Python bindings if they aren't yet
  available (defaults to `FALSE`).

## Value

Logical indicating whether Python is initialized.

## Note

The `py_numpy_available` function is a superset of the `py_available`
function (it calls `py_available` first before checking for NumPy).
