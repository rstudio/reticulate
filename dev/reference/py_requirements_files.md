# Write and read Python requirements files

- `py_write_requirements()` writes the requirements currently tracked by
  [`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md).
  If `freeze = TRUE` or if the `python` environment is not ephemeral, it
  writes a fully resolved manifest via `pip freeze`.

- `py_read_requirements()` reads `requirements.txt` and
  `.python-version`, and applies them with
  [`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md).
  By default, entries are added (`action = "add"`).

These are primarily an alternative interface to
[`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md),
but can also work with non-ephemeral virtual environments.

## Usage

``` r
py_write_requirements(
  packages = "requirements.txt",
  python_version = ".python-version",
  ...,
  freeze = NULL,
  python = py_exe(),
  quiet = FALSE
)

py_read_requirements(
  packages = "requirements.txt",
  python_version = ".python-version",
  ...,
  action = c("add", "set", "remove", "none")
)
```

## Arguments

- packages:

  Path to the package requirements file. Defaults to
  `"requirements.txt"`. Use `NULL` to skip.

- python_version:

  Path to the Python version file. Defaults to `".python-version"`. Use
  `NULL` to skip.

- ...:

  Unused; must be empty.

- freeze:

  Logical. If `TRUE`, writes a fully resolved list of installed packages
  using `pip freeze`. If `FALSE`, writes only the requirements tracked
  by
  [`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md).

- python:

  Path to the Python executable to use.

- quiet:

  Logical; if `TRUE`, suppresses the informational messages that print
  `wrote '<path>'` for each file written.

- action:

  How to apply requirements read by `py_read_requirements()`: `"add"`
  (default) adds to existing requirements, `"set"` replaces them,
  `"remove"` removes matching entries, or `"none"` skips applying them
  and returns the read values.

## Value

Invisibly, a list with two named elements:

- `packages`:

  Character vector of package requirements.

- `python_version`:

  String specifying the Python version.

To get just the return value without writing any files, you can pass
`NULL` for file paths, like this:

    py_write_requirements(NULL, NULL)
    py_write_requirements(NULL, NULL, freeze = TRUE)

## Note

To continue using
[`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md)
locally while keeping a `requirements.txt` up-to-date for deployments,
you can register an exit handler in `.Rprofile` like this:

    reg.finalizer(
      asNamespace("reticulate"),
      function(ns) {
        if (
          reticulate::py_available() &&
            isTRUE(reticulate::py_config()$ephemeral)
        ) {
          reticulate::py_write_requirements(quiet = TRUE)
        }
      },
      onexit = TRUE
    )

This approach is only recommended if you are using `git`.

Alternatively, you can transition away from using ephemeral python
environemnts via
[`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md)
to using a persistent local virtual environment you manage. You can
create a local virtual environment from `requirements.txt` and
`.python-version` using
[`virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md):

    # Note: '.venv' in the current directory is auto-discovered by reticulate.
    # https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery
    virtualenv_create(
      "./.venv",
      version = readLines(".python-version"),
      requirements = "requirements.txt"
    )

If you run into issues, be aware that `requirements.txt` and
`.python-version` may not contain all the information necessary to
reproduce the Python environment if the R code sets environment
variables like `UV_INDEX` or `UV_CONSTRAINT`.
