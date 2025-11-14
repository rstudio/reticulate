# Python configuration

Retrieve information about the version of Python currently being used by
`reticulate`.

## Usage

``` r
py_config()
```

## Value

Information about the version of Python in use, as an R list with class
`"py_config"`.

## Details

If Python has not yet been initialized, then calling `py_config()` will
force the initialization of Python. See
[`py_discover_config()`](https://rstudio.github.io/reticulate/reference/py_discover_config.md)
for more details.
