# Conda Tools

Tools for managing Python `conda` environments.

## Usage

``` r
conda_list(conda = "auto")

conda_create(
  envname = NULL,
  packages = NULL,
  ...,
  forge = TRUE,
  channel = character(),
  environment = NULL,
  conda = "auto",
  python_version = miniconda_python_version(),
  additional_create_args = character()
)

conda_clone(envname, ..., clone = "base", conda = "auto")

conda_export(
  envname,
  file = if (json) "environment.json" else "environment.yml",
  json = FALSE,
  ...,
  conda = "auto"
)

conda_remove(envname, packages = NULL, conda = "auto")

conda_install(
  envname = NULL,
  packages,
  forge = TRUE,
  channel = character(),
  pip = FALSE,
  pip_options = character(),
  pip_ignore_installed = FALSE,
  conda = "auto",
  python_version = NULL,
  additional_create_args = character(),
  additional_install_args = character(),
  ...
)

conda_binary(conda = "auto")

conda_exe(conda = "auto")

conda_version(conda = "auto")

conda_update(conda = "auto")

conda_python(envname = NULL, conda = "auto", all = FALSE)

conda_search(
  matchspec,
  forge = TRUE,
  channel = character(),
  conda = "auto",
  ...
)

condaenv_exists(envname = NULL, conda = "auto")
```

## Arguments

- conda:

  The path to a `conda` executable. Use `"auto"` to allow `reticulate`
  to automatically find an appropriate `conda` binary. See **Finding
  Conda** and `conda_binary()` for more details.

- envname:

  The name of, or path to, a conda environment.

- packages:

  A character vector, indicating package names which should be installed
  or removed. Use `<package>==<version>` to request the installation of
  a specific version of a package. A `NULL` value for `conda_remove()`
  will be interpretted to `"--all"`, removing the entire environment.

- ...:

  Optional arguments, reserved for future expansion.

- forge:

  Boolean; include the [conda-forge](https://conda-forge.org/)
  repository?

- channel:

  An optional character vector of conda channels to include. When
  specified, the `forge` argument is ignored. If you need to specify
  multiple channels, including the conda forge, you can use
  `c("conda-forge", <other channels>)`.

- environment:

  The path to an environment definition, generated via (for example)
  `conda_export()`, or via `conda env export`. When provided, the conda
  environment will be created using this environment definition, and
  other arguments will be ignored.

- python_version:

  The version of Python to be installed. Set this if you'd like to
  change the version of Python associated with a particular conda
  environment.

- additional_create_args:

  An optional character vector of additional arguments to use in the
  call to `conda create`.

- clone:

  The name of the conda environment to be cloned.

- file:

  The path where the conda environment definition will be written.

- json:

  Boolean; should the environment definition be written as JSON? By
  default, conda exports environments as YAML.

- pip:

  Boolean; use `pip` for package installation? By default, packages are
  installed from the active conda channels.

- pip_options:

  An optional character vector of additional command line arguments to
  be passed to `pip`. Only relevant when `pip = TRUE`.

- pip_ignore_installed:

  Ignore already-installed versions when using pip? (defaults to
  `FALSE`). Set this to `TRUE` so that specific package versions can be
  installed even if they are downgrades. The `FALSE` option is useful
  for situations where you don't want a pip install to attempt an
  overwrite of a conda binary package (e.g. SciPy on Windows which is
  very difficult to install via pip due to compilation requirements).

- additional_install_args:

  An optional character vector of additional arguments to use in the
  call to `conda install`.

- all:

  Boolean; report all instances of Python found?

- matchspec:

  A conda MatchSpec query string.

## Value

`conda_list()` returns an R `data.frame`, with `name` giving the name of
the associated environment, and `python` giving the path to the Python
binary associated with that environment.

`conda_create()` returns the path to the Python binary associated with
the newly-created conda environment.

`conda_clone()` returns the path to Python within the newly-created
conda environment.

`conda_export()` returns the path to the exported environment
definition, invisibly.

`conda_search()` returns an R `data.frame` describing packages that
matched against `matchspec`. The data frame will usually include fields
`name` giving the package name, `version` giving the package version,
`build` giving the package build, and `channel` giving the channel the
package is hosted on.

## Finding Conda

Most of `reticulate`'s conda APIs accept a `conda` parameter, used to
control the `conda` binary used in their operation. When
`conda = "auto"`, `reticulate` will attempt to automatically find a
conda installation. The following locations are searched, in order:

1.  The location specified by the `reticulate.conda_binary` R option,

2.  The location specified by the `RETICULATE_CONDA` environment
    variable,

3.  The
    [`miniconda_path()`](https://rstudio.github.io/reticulate/reference/miniconda_path.md)
    location (if it exists),

4.  The program `PATH`,

5.  A set of pre-defined locations where conda is typically installed.

To force `reticulate` to use a particular `conda` binary, we recommend
setting:

    options(reticulate.conda_binary = "/path/to/conda")

This can be useful if your conda installation lives in a location that
`reticulate` is unable to automatically discover.

## See also

[`conda_run2()`](https://rstudio.github.io/reticulate/reference/conda_run2.md)
