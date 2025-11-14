# Python version

Get the version of Python currently being used by `reticulate`.

## Usage

``` r
py_version(patch = FALSE)
```

## Arguments

- patch:

  boolean, whether to include the patch level in the returned version.

## Value

The version of Python currently used, or `NULL` if Python has not yet
been initialized by `reticulate`.
