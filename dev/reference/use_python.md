# Use Python

Manually select the version of Python to be used by `reticulate`.

Note that beginning with Reticulate version 1.41, manually selecting a
Python installation is generally not necessary, as reticulate is able to
automatically resolve an ephemeral Python environment with all necessary
Python requirements declared via
[`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md).

## Usage

``` r
use_python(python, required = NULL)

use_python_version(version, required = NULL)

use_virtualenv(virtualenv = NULL, required = NULL)

use_condaenv(condaenv = NULL, conda = "auto", required = NULL)

use_miniconda(condaenv = NULL, required = NULL)
```

## Arguments

- python:

  The path to a Python binary.

- required:

  Is the requested copy of Python required? If `TRUE`, an error will be
  emitted if the requested copy of Python does not exist. If `FALSE`,
  the request is taken as a hint only, and scanning for other versions
  will still proceed. A value of `NULL` (the default), is equivalent to
  `TRUE`.

- version:

  The version of Python to use. `reticulate` will search for versions of
  Python as installed by the
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  helper function.

- virtualenv:

  Either the name of, or the path to, a Python virtual environment.

- condaenv:

  The conda environment to use. For `use_condaenv()`, this can be the
  name, the absolute prefix path, or the absolute path to the python
  binary. If the name is ambiguous, the first environment is used and a
  warning is issued. For `use_miniconda()`, the only conda installation
  searched is the one installed by
  [`install_miniconda()`](https://rstudio.github.io/reticulate/dev/reference/install_miniconda.md).

- conda:

  The path to a `conda` executable. By default, `reticulate` will check
  the `PATH`, as well as other standard locations for Anaconda
  installations.

## Details

The `reticulate` package initializes its Python bindings lazily – that
is, it does not initialize its Python bindings until an API that
explicitly requires Python to be loaded is called. This allows users and
package authors to request particular versions of Python by calling
`use_python()` or one of the other helper functions documented in this
help file.

## RETICULATE_PYTHON

The `RETICULATE_PYTHON` environment variable can also be used to control
which copy of Python `reticulate` chooses to bind to. It should be set
to the path to a Python interpreter, and that interpreter can either be:

- A standalone system interpreter,

- Part of a virtual environment,

- Part of a Conda environment.

When set, this will override any other requests to use a particular copy
of Python. Setting this in `~/.Renviron` (or optionally, a project
`.Renviron`) can be a useful way of forcing `reticulate` to use a
particular version of Python.

## Caveats

Note that the requests for a particular version of Python via
`use_python()` and friends only persist for the active session; they
must be re-run in each new R session as appropriate.

If `use_python()` (or one of the other `use_*()` functions) are called
multiple times, the most recently-requested version of Python will be
used. Note that any request to `use_python()` will always be overridden
by the `RETICULATE_PYTHON` environment variable, if set.

The
[`py_config()`](https://rstudio.github.io/reticulate/dev/reference/py_config.md)
function will also provide a short note describing why `reticulate`
chose to select the version of Python that was ultimately activated.
