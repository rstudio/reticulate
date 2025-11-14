# Convert Python bytes to an R character or raw vector

Convert Python bytes to an R character or raw vector

## Usage

``` r
# S3 method for class 'python.builtin.bytes'
as.character(
  x,
  encoding = "utf-8",
  errors = "strict",
  nul = stop("Embedded NUL in string."),
  ...
)

# S3 method for class 'python.builtin.bytes'
as.raw(x)
```

## Arguments

- x:

  object to be coerced or tested.

- encoding:

  Encoding to use for conversion (defaults to utf-8)

- errors:

  Policy for handling conversion errors. Default is 'strict' which
  raises an error. Other possible values are 'ignore' and 'replace'.

- nul:

  Action to take if the bytes contain an embedded NUL (`\x00`). Python
  allows embedded `NUL`s in strings, while R does not. There are four
  options for handling embedded `NUL`s:

  1.  Error: This is the default

  2.  Replace: Supply a replacement string: `nul = "<NUL>"`

  3.  Remove: Supply an empty string: `nul = ""`

  4.  Split: Supply an R `NULL` to indicate that string should be split
      at embedded `NUL` bytes: `nul = NULL`

- ...:

  further arguments passed to or from other methods.

## See also

[`as.character.python.builtin.str()`](https://rstudio.github.io/reticulate/dev/reference/as.character.python.builtin.str.md)

## Examples

``` r
if (FALSE) { # reticulate::py_available()
# A bytes object with embedded NULs
b <- import_builtins(convert = FALSE)$bytes(
  as.raw(c(0x61, 0x20, 0x62, 0x00, 0x63, 0x20, 0x64)) # "a b<NUL>c d"
)

try(as.character(b))            # Error : Embedded NUL in string.
as.character(b, nul = "<NUL>")  # Replace: "a b<NUL>c d"
as.character(b, nul = "")       # Remove: "a bc d"
as.character(b, nul = NULL)     # Split: "a b" "c d"
}
```
