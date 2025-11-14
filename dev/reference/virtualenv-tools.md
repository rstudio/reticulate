# Interface to Python Virtual Environments

R functions for managing Python [virtual
environments](https://virtualenv.pypa.io/en/stable/).

## Usage

``` r
virtualenv_create(
  envname = NULL,
  python = virtualenv_starter(version),
  ...,
  version = NULL,
  packages = "numpy",
  requirements = NULL,
  force = FALSE,
  module = getOption("reticulate.virtualenv.module"),
  system_site_packages = getOption("reticulate.virtualenv.system_site_packages", default
    = FALSE),
  pip_version = getOption("reticulate.virtualenv.pip_version", default = NULL),
  setuptools_version = getOption("reticulate.virtualenv.setuptools_version", default =
    NULL),
  extra = getOption("reticulate.virtualenv.extra", default = NULL)
)

virtualenv_install(
  envname = NULL,
  packages = NULL,
  ignore_installed = FALSE,
  pip_options = character(),
  requirements = NULL,
  ...,
  python_version = NULL
)

virtualenv_remove(envname = NULL, packages = NULL, confirm = interactive())

virtualenv_list()

virtualenv_root()

virtualenv_python(envname = NULL)

virtualenv_exists(envname = NULL)

virtualenv_starter(version = NULL, all = FALSE)
```

## Arguments

- envname:

  The name of, or path to, a Python virtual environment. If this name
  contains any slashes, the name will be interpreted as a path; if the
  name does not contain slashes, it will be treated as a virtual
  environment within `virtualenv_root()`. When `NULL`, the virtual
  environment as specified by the `RETICULATE_PYTHON_ENV` environment
  variable will be used instead. To refer to a virtual environment in
  the current working directory, you can prefix the path with
  `./<name>`.

- python:

  The path to a Python interpreter, to be used with the created virtual
  environment. This can also accept a version constraint like `"3.10"`,
  which is passed on to `virtualenv_starter()` to find a suitable python
  binary.

- ...:

  Optional arguments; currently ignored and reserved for future
  expansion.

- version, python_version:

  (string) The version of Python to use when creating a virtual
  environment. Python installations will be searched for using
  `virtualenv_starter()`. This can a specific version, like `"3.9"` or
  `"3.9.3"`, or a comma separated list of version constraints, like
  `">=3.8"`, or `"<=3.11,!=3.9.3,>3.6"`

- packages:

  A set of Python packages to install (via `pip install`) into the
  virtual environment, after it has been created. By default, the
  `"numpy"` package will be installed, and the `pip`, `setuptools` and
  `wheel` packages will be updated. Set this to `FALSE` to avoid
  installing any packages after the virtual environment has been
  created.

- requirements:

  Filepath to a pip requirements file.

- force:

  Boolean; force recreating the environment specified by `envname`, even
  if it already exists. If `TRUE`, the pre-existing environment is first
  deleted and then recreated. Otherwise, if `FALSE` (the default), the
  path to the existing environment is returned.

- module:

  The Python module to be used when creating the virtual environment –
  typically, `virtualenv` or `venv`. When `NULL` (the default), `venv`
  will be used if available with Python \>= 3.6; otherwise, the
  `virtualenv` module will be used.

- system_site_packages:

  Boolean; create new virtual environments with the
  `--system-site-packages` flag, thereby allowing those virtual
  environments to access the system's site packages? Defaults to
  `FALSE`.

- pip_version:

  The version of `pip` to be installed in the virtual environment.
  Relevant only when `module == "virtualenv"`. Set this to `FALSE` to
  disable installation of `pip` altogether.

- setuptools_version:

  The version of `setuptools` to be installed in the virtual
  environment. Relevant only when `module == "virtualenv"`. Set this to
  `FALSE` to disable installation of `setuptools` altogether.

- extra:

  An optional set of extra command line arguments to be passed.
  Arguments should be quoted via
  [`shQuote()`](https://rdrr.io/r/base/shQuote.html) when necessary.

- ignore_installed:

  Boolean; ignore previously-installed versions of the requested
  packages? (This should normally be `TRUE`, so that pre-installed
  packages available in the site libraries are ignored and hence
  packages are installed into the virtual environment.)

- pip_options:

  An optional character vector of additional command line arguments to
  be passed to `pip`.

- confirm:

  Boolean; confirm before removing packages or virtual environments?

- all:

  If `TRUE`, `virtualenv_starter()` returns a 2-column data frame, with
  column names `path` and `version`. If `FALSE`, only a single path to a
  python binary is returned, corresponding to the first entry when
  `all = TRUE`, or `NULL` if no suitable python binaries were found.

## Details

Virtual environments are by default located at `~/.virtualenvs`
(accessed with the `virtualenv_root()` function). You can change the
default location by defining the `RETICULATE_VIRTUALENV_ROOT` or
`WORKON_HOME` environment variables.

Virtual environments are created from another "starter" or "seed" Python
already installed on the system. Suitable Pythons installed on the
system are found by `virtualenv_starter()`.
