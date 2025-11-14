# List installed Python packages

List the Python packages that are installed in the requested Python
environment.

## Usage

``` r
py_list_packages(
  envname = NULL,
  type = c("auto", "virtualenv", "conda"),
  python = NULL
)
```

## Arguments

- envname:

  The name of, or path to, a Python virtual environment. Ignored when
  `python` is non-`NULL`.

- type:

  The virtual environment type. Useful if you have both virtual
  environments and Conda environments of the same name on your system,
  and you need to disambiguate them.

- python:

  The path to a Python executable.

## Value

An R data.frame, with columns:

- `package`:

  The package name.

- `version`:

  The package version.

- `requirement`:

  The package requirement.

- `channel`:

  (Conda only) The channel associated with this package.

## Details

When `envname` is `NULL`, `reticulate` will use the "default" version of
Python, as reported by
[`py_exe()`](https://rstudio.github.io/reticulate/reference/py_exe.md).
This implies that you can call `py_list_packages()` without arguments in
order to list the installed Python packages in the version of Python
currently used by `reticulate`.
