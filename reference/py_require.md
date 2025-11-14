# Declare Python Requirements

`py_require()` allows you to declare Python requirements for the R
session, including Python packages, any version constraints on those
packages, and any version constraints on Python itself. Reticulate can
then automatically create and use an ephemeral Python environment that
satisfies all these requirements.

## Usage

``` r
py_require(
  packages = NULL,
  python_version = NULL,
  ...,
  exclude_newer = NULL,
  action = c("add", "remove", "set")
)
```

## Arguments

- packages:

  A character vector of Python packages to be available during the
  session. These can be simple package names like `"jax"` or names with
  version constraints like `"jax[cpu]>=0.5"`. Pip style syntax for
  installing from local files or a git repository is also supported (see
  details).

- python_version:

  A character vector of Python version constraints  
  (e.g., `"3.10"` or `">=3.9,<3.13"`).

- ...:

  Reserved for future extensions; must be empty.

- exclude_newer:

  Limit package versions to those published before a specified date.
  This offers a lightweight alternative to freezing package versions,
  helping guard against Python package updates that break a workflow.
  Accepts strings formatted as RFC 3339 timestamps (e.g.,
  `"2006-12-02T02:07:43Z"`) and local dates in the same format (e.g.,
  `"2006-12-02"`) in your system's configured time zone. Once
  `exclude_newer` is set, only the `set` action can override it.

- action:

  Determines how `py_require()` processes the provided requirements.
  Options are:

  - `"add"` (the default): Adds the entries to the current set of
    requirements.

  - `"remove"`: Removes *exact* matches from the requirements list.
    Requests to remove nonexistent entries are ignored. For example, if
    `"numpy==2.2.2"` is in the list, passing `"numpy"` with
    `action="remove"` will not remove it.

  - `"set"`: Clears all existing requirements and replaces them with the
    provided ones. Packages and the Python version can be set
    independently.

## Value

`py_require()` is primarily called for its side effect of modifying the
manifest of "Python requirements" for the current R session that
reticulate maintains internally. `py_require()` usually returns `NULL`
invisibly. If `py_require()` is called with no arguments, it returns the
current manifest–a list with names `packages`, `python_version`, and
`exclude_newer.` The list also has a class attribute, to provide a print
method.

## Details

Reticulate will only use an ephemeral environment if no other Python
installation is found earlier in the [Order of
Discovery](https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery).
You can also force reticulate to use an ephemeral environment by setting
`Sys.setenv(RETICULATE_PYTHON="managed")`, or you can disable reticulate
from using an ephemeral environment by setting
`Sys.setenv(RETICULATE_USE_MANAGED_VENV="no")`.

The ephemeral virtual environment is not created until the user
interacts with Python for the first time in the R session, typically
when
[`import()`](https://rstudio.github.io/reticulate/reference/import.md)
is first called.

If `py_require()` is called with new requirements after reticulate has
already initialized an ephemeral Python environment, a new ephemeral
environment is activated on top of the existing one. Once Python is
initialized, only adding packages is supported—removing packages,
changing the Python version, or modifying `exclude_newer` is not
possible.

Calling `py_require()` without arguments returns a list of the currently
declared requirements.

R packages can also call `py_require()` (e.g., in `.onLoad()` or
elsewhere) to declare Python dependencies. The print method for
`py_require()` displays the Python dependencies declared by R packages
in the current session.

## Note

Reticulate uses [`uv`](https://docs.astral.sh/uv/) to resolve Python
dependencies. Many `uv` options can be customized via environment
variables, as described
[here](https://docs.astral.sh/uv/configuration/environment/). For
example:

- If temporarily offline, to resolve packages from cache without
  checking for updates, set:  
  `Sys.setenv(UV_OFFLINE = "1")`.

- To use an additional package index:  
  `Sys.setenv(UV_INDEX = "https://download.pytorch.org/whl/cpu")`.  
  (To add multiple additional indexes, `UV_INDEX` can be a list of
  space-separated urls).

- To change the default package index:  
  `Sys.setenv(UV_DEFAULT_INDEX = "https://my.org/python-packages-index/")`

- To allow resolving a prerelease dependency:  
  `Sys.setenv(UV_PRERELEASE = "allow")`.

- To force `uv` to create ephemeral environments using the system
  python:  
  `Sys.setenv(UV_PYTHON_PREFERENCE = "only-system")`

For more advanced customization needs, there’s also the option to
configure `uv` with a user-level or system-level `uv.toml` file.

### Installing from alternate sources

The `packages` argument also supports declaring a dependency from a Git
repository or a local file. Below are some examples of valid `packages`
strings:

- Install Ruff from a specific Git tag:

      "git+https://github.com/astral-sh/ruff@v0.2.0"

- Install Ruff from a specific Git commit:

      "git+https://github.com/astral-sh/ruff@1fadefa67b26508cc59cf38e6130bde2243c929d"

- Install Ruff from a specific Git branch:

      "git+https://github.com/astral-sh/ruff@main"

- Install MarkItDown from the `main` branch—find the package in the
  subdirectory 'packages/markitdown':

      "markitdown@git+https://github.com/microsoft/markitdown.git@main#subdirectory=packages/markitdown"

- Install MarkItDown from the local filesystem by providing an absolute
  path to a directory containing a `pyproject.toml` or `setup.py` file:

      "markitdown@/Users/tomasz/github/microsoft/markitdown/packages/markitdown/"

See more examples
[here](https://docs.astral.sh/uv/pip/packages/#installing-a-package) and
[here](https://pip.pypa.io/en/stable/cli/pip_install/#examples).

### Clearing the Cache

If `uv` is already installed on your machine, `reticulate` will use the
existing `uv` installation as-is, including its default `cache dir`
location. To clear the caches of a self-managed `uv` installation, send
the following system commands to `uv`:

    uv cache clean
    rm -r "$(uv python dir)"
    rm -r "$(uv tool dir)"

If an existing installation of `uv` is not found, `reticulate` will
automatically download and store it, along with other downloaded
artifacts and ephemeral environments, in the
`tools::R_user_dir("reticulate", "cache")` directory. To clear this
cache manually, delete the directory:

    # delete uv, ephemeral virtual environments, and all downloaded artifacts
    unlink(tools::R_user_dir("reticulate", "cache"), recursive = TRUE)

Reticulate also clears its managed cache automatically on an interval,
defaulting to every 120 days. Configure this interval in `.Rprofile`
with:

    options(reticulate.max_cache_age = as.difftime(30, units = "days"))
