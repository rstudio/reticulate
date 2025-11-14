# Configure a Python Environment

Configure a Python environment, satisfying the Python dependencies of
any loaded R packages.

## Usage

``` r
configure_environment(package = NULL, force = FALSE)
```

## Arguments

- package:

  The name of a package to configure. When `NULL`, `reticulate` will
  instead look at all loaded packages and discover their associated
  Python requirements.

- force:

  Boolean; force configuration of the Python environment? Note that
  `configure_environment()` is a no-op within non-interactive R
  sessions. Use this if you require automatic environment configuration,
  e.g. when testing a package on a continuous integration service.

## Details

Normally, this function should only be used by package authors, who want
to ensure that their package dependencies are installed in the active
Python environment. For example:

    .onLoad <- function(libname, pkgname) {
      reticulate::configure_environment(pkgname)
    }

If the Python session has not yet been initialized, or if the user is
not using the default Miniconda Python installation, no action will be
taken. Otherwise, `reticulate` will take this as a signal to install any
required Python dependencies into the user's Python environment.

If you'd like to disable `reticulate`'s auto-configure behavior
altogether, you can set the environment variable:

    RETICULATE_AUTOCONFIGURE = FALSE

e.g. in your `~/.Renviron` or similar.

Note that, in the case where the Python session has not yet been
initialized, `reticulate` will automatically ensure your required Python
dependencies are installed after the Python session is initialized (when
appropriate).
