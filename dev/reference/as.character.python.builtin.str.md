# Convert a Python string to an R Character Vector

Convert a Python string to an R Character Vector

## Usage

``` r
# S3 method for class 'python.builtin.str'
as.character(x, nul = stop("Embedded NUL in string."), ...)
```

## Arguments

- x:

  A Python string

- nul:

  Action to take if the Python string contains an embedded NUL (`\x00`).
  Python allows embedded `NUL`s in strings, while R does not. There are
  four options for handling embedded `NUL`s:

  1.  Error: This is the default

  2.  Replace: Supply a replacement string: `nul = "<NUL>"`

  3.  Remove: Supply an empty string: `nul = ""`

  4.  Split: Supply an R `NULL` to indicate that string should be split
      at embedded `NUL` bytes: `nul = NULL`

- ...:

  Unused

## Value

An R character vector. The returned vector will always of length 1,
unless `nul = NULL` was supplied.

## Examples

``` r
if (FALSE) { # reticulate::py_available()
# Given a Python function that errors when it attempts to return
# a string with an embedded NUL
py_run_string('
def get_string_w_nul():
   return "a b" + chr(0) + "c d"
')
get_string_w_nul <- py$get_string_w_nul

try(get_string_w_nul()) # Error : Embedded NUL in string.

# To get the string into R, use `r_to_py()` on the function to stop it from
# eagerly converting the Python string to R, and then call `as.character()` with
# a `nul` argument supplied to convert the string to R.
get_string_w_nul <- r_to_py(get_string_w_nul)
get_string_w_nul() # unconverted python string: inherits(x, 'python.builtin.str')
as.character(get_string_w_nul(), nul = "<NUL>")  # Replace: "a b<NUL>c d"
as.character(get_string_w_nul(), nul = "")       # Remove: "a bc d"
as.character(get_string_w_nul(), nul = NULL)     # Split: "a b" "c d"

# cleanup example
rm(get_string_w_nul); py$get_string_w_nul <- NULL
}
```
