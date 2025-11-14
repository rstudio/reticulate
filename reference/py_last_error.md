# Get or (re)set the last Python error encountered.

Get or (re)set the last Python error encountered.

## Usage

``` r
py_clear_last_error()

py_last_error(exception)
```

## Arguments

- exception:

  A python exception object. If provided, the provided exception is set
  as the last exception.

## Value

For `py_last_error()`, `NULL` if no error has yet been encountered.
Otherwise, a named list with entries:

- `"type"`: R string, name of the exception class.

- `"value"`: R string, formatted exception message.

- `"traceback"`: R character vector, the formatted python traceback,

- `"message"`: The full formatted raised exception, as it would be
  printed in Python. Includes the traceback, type, and value.

- `"r_trace"`: A `data.frame` with class `rlang_trace` and columns:

  - `call`: The R callstack, `full_call`, summarized for pretty
    printing.

  - `full_call`: The R callstack. (Output of
    [`sys.calls()`](https://rdrr.io/r/base/sys.parent.html) at the error
    callsite).

  - `parent`: The parent of each frame in callstack. (Output of
    [`sys.parents()`](https://rdrr.io/r/base/sys.parent.html) at the
    error callsite).

  - Additional columns for internals use: `namespace`, `visible`,
    `scope`.

And attribute `"exception"`, a `'python.builtin.Exception'` object.

The named list has `class` `"py_error"`, and has a default `print`
method that is the equivalent of `cat(py_last_error()$message)`.

## Examples

``` r
if (FALSE) { # \dontrun{

# see last python exception with R traceback
reticulate::py_last_error()

# see the full R callstack from the last Python exception
reticulate::py_last_error()$r_trace$full_call

# run python code that might error,
# without modifying the user-visible python exception

safe_len <- function(x) {
  last_err <- py_last_error()
  tryCatch({
    # this might raise a python exception if x has no `__len__` method.
    import_builtins()$len(x)
  }, error = function(e) {
    # py_last_error() was overwritten, is now "no len method for 'object'"
    py_last_error(last_err) # restore previous exception
    -1L
  })
}

safe_len(py_eval("object"))
} # }
```
