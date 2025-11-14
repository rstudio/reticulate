# Run a Python REPL

This function provides a Python REPL in the R session, which can be used
to interactively run Python code. All code executed within the REPL is
run within the Python main module, and any generated Python objects will
persist in the Python session after the REPL is detached.

## Usage

``` r
repl_python(
  module = NULL,
  quiet = getOption("reticulate.repl.quiet", default = FALSE),
  input = NULL
)
```

## Arguments

- module:

  An (optional) Python module to be imported before the REPL is
  launched.

- quiet:

  Boolean; print a startup banner when launching the REPL? If `TRUE`,
  the banner will be suppressed.

- input:

  Python code to be run within the REPL. Setting this can be useful if
  you'd like to drive the Python REPL programmatically.

## Details

When working with R and Python scripts interactively, one can activate
the Python REPL with `repl_python()`, run Python code, and later run
`exit` to return to the R console.

## Magics

A handful of magics are supported in `repl_python()`:

Lines prefixed with `!` are executed as system commands:

- `!cmd --arg1 --arg2`: Execute arbitrary system commands

Magics start with a `%` prefix. Supported magics include:

- `%conda ...` executes a conda command in the active conda environment

- `%pip ...` executes pip for the active python.

- `%load`, `%loadpy`, `%run` executes a python file.

- `%system`, `!!` executes a system command and capture output

- `%env`: read current environment variables.

  - `%env name`: read environment variable 'name'.

  - `%env name=val`, `%env name val`: set environment variable 'name' to
    'val'. `val` elements in [`{}`](https://rdrr.io/r/base/Paren.html)
    are interpolated using f-strings (required Python \>= 3.6).

- `%cd <dir>` change working directory.

  - `%cd -`: change to previous working directory (as set by `%cd`).

  - `%cd -3`: change to 3rd most recent working directory (as set by
    `%cd`).

  - `%cd -foo/bar`: change to most recent working directory matching
    `"foo/bar"` regex (in history of directories set via `%cd`).

- `%pwd`: print current working directory.

- `%dhist`: print working directory history.

Additionally, the output of system commands can be captured in a
variable, e.g.:

- `x = !ls`

where `x` will be a list of strings, consisting of stdout output split
in `"\n"` (stderr is not captured).

## Example

    # enter the Python REPL, create a dictionary, and exit
    repl_python()
    dictionary = {'alpha': 1, 'beta': 2}
    exit

    # access the created dictionary from R
    py$dictionary
    # $alpha
    # [1] 1
    #
    # $beta
    # [1] 2

## See also

[py](https://rstudio.github.io/reticulate/dev/reference/py.md), for
accessing objects created using the Python REPL.
