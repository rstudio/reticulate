# Discover the version of Python to use with reticulate.

This function enables callers to check which versions of Python will be
discovered on a system as well as which one will be chosen for use with
reticulate.

## Usage

``` r
py_discover_config(required_module = NULL, use_environment = NULL)
```

## Arguments

- required_module:

  A optional module name that will be used to select the Python
  environment used.

- use_environment:

  An optional virtual/conda environment name to prefer in the search.

## Value

Python configuration object.

## Details

The order of discovery is documented in
[`vignette("versions")`](https://rstudio.github.io/reticulate/dev/articles/versions.md),
also available online
[here](https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery)
