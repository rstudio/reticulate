# Run a command in a conda environment

This function runs a command in a chosen conda environment.

## Usage

``` r
conda_run2(
  cmd,
  args = c(),
  conda = "auto",
  envname = NULL,
  cmd_line = paste(shQuote(cmd), paste(args, collapse = " ")),
  intern = FALSE,
  echo = !intern
)
```

## Arguments

- cmd:

  The system command to be invoked, as a character string.

- args:

  A character vector of arguments to the command. The arguments should
  be quoted e.g. by [`shQuote()`](https://rdrr.io/r/base/shQuote.html)
  in case they contain space or other special characters (a double quote
  or backslash on Windows, shell-specific special characters on Unix).

- conda:

  The path to a `conda` executable. Use `"auto"` to allow `reticulate`
  to automatically find an appropriate `conda` binary. See **Finding
  Conda** and
  [`conda_binary()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  for more details.

- envname:

  The name of, or path to, a conda environment.

- cmd_line:

  The command line to be executed, as a character string. This is
  automatically generated from `cmd` and `args`, but can be provided
  directly if needed (if provided, it overrides `cmd` and `args`).

- intern:

  A logical (not `NA`) which indicates whether to capture the output of
  the command as an R character vector. If `FALSE` (the default), the
  return value is the error code (`0` for success).

- echo:

  A logical (not `NA`) which indicates whether to echo the command to
  the console before running it.

## Value

`conda_run2()` runs a command in the desired conda environment. If
`intern = TRUE` the output is returned as a character vector; if
`intern = FALSE` (the deafult), then the return value is the error code
(0 for success). See [`shell()`](https://rdrr.io/r/base/system.html) (on
windows) or [`system2()`](https://rdrr.io/r/base/system2.html) on macOS
or Linux for more details.

## Details

Note that, whilst the syntax is similar to
[`system2()`](https://rdrr.io/r/base/system2.html), the function
dynamically generates a shell script with commands to activate the
chosen conda environent. This avoids issues with quoting, as discussed
in this [GitHub issue](https://github.com/conda/conda/issues/10972).

## See also

[`conda-tools`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
