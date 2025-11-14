# [Deprecated](https://rdrr.io/r/base/Deprecated.html) Create a Python function that will always be called on the main thread

Beginning with reticulate v1.39.0, every R function is a "main thread
func". Usage of `py_main_thread_func()` is no longer necessary.

## Usage

``` r
py_main_thread_func(f)
```

## Arguments

- f:

  An R function with arbitrary arguments

## Value

A Python function that delegates to the passed R function, which is
guaranteed to always be called on the main thread.

## Details

This function is helpful when you need to provide a callback to a Python
library which may invoke the callback on a background thread. As R
functions must run on the main thread, wrapping the R function with
`py_main_thread_func()` will ensure that R code is only executed on the
main thread.
