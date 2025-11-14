# Install Python packages

Install Python packages into a virtual environment or Conda environment.

## Usage

``` r
py_install(
  packages,
  envname = NULL,
  method = c("auto", "virtualenv", "conda"),
  conda = "auto",
  python_version = NULL,
  pip = FALSE,
  ...,
  pip_ignore_installed = ignore_installed,
  ignore_installed = FALSE
)
```

## Arguments

- packages:

  A vector of Python packages to install.

- envname:

  The name, or full path, of the environment in which Python packages
  are to be installed. When `NULL` (the default), the active environment
  as set by the `RETICULATE_PYTHON_ENV` variable will be used; if that
  is unset, then the `r-reticulate` environment will be used.

- method:

  Installation method. By default, "auto" automatically finds a method
  that will work in the local environment. Change the default to force a
  specific installation method. Note that the "virtualenv" method is not
  available on Windows.

- conda:

  The path to a `conda` executable. Use `"auto"` to allow `reticulate`
  to automatically find an appropriate `conda` binary. See **Finding
  Conda** and
  [`conda_binary()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  for more details.

- python_version:

  The requested Python version. Ignored when attempting to install with
  a Python virtual environment.

- pip:

  Boolean; use `pip` for package installation? This is only relevant
  when Conda environments are used, as otherwise packages will be
  installed from the Conda repositories.

- ...:

  Additional arguments passed to
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  or
  [`virtualenv_install()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md).

- pip_ignore_installed, ignore_installed:

  Boolean; whether pip should ignore previously installed versions of
  the requested packages. Setting this to `TRUE` causes pip to install
  the latest versions of all dependencies into the requested
  environment. This ensure that no dependencies are satisfied by a
  package that exists either in the site library or was previously
  installed from a different–potentially incompatible–distribution
  channel. (`ignore_installed` is an alias for `pip_ignore_installed`,
  `pip_ignore_installed` takes precedence).

## Details

On Linux and OS X the "virtualenv" method will be used by default
("conda" will be used if virtualenv isn't available). On Windows, the
"conda" method is always used.

## See also

[`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md),
for installing packages into conda environments.
[`virtualenv_install()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md),
for installing packages into virtual environments.
