# Install Python

Download and install Python, using the
[pyenv](https://github.com/pyenv/pyenv). and
[pyenv-win](https://github.com/pyenv-win/pyenv-win) projects.

## Usage

``` r
install_python(
  version = "3.11:latest",
  list = FALSE,
  force = FALSE,
  optimized = TRUE
)
```

## Arguments

- version:

  The version of Python to install.

- list:

  Boolean; if set, list the set of available Python versions?

- force:

  Boolean; force re-installation even if the requested version of Python
  is already installed?

- optimized:

  Boolean; if `TRUE`, installation will take significantly longer but
  should result in a faster Python interpreter. Only applicable on macOS
  and Linux.

## Details

In general, it is recommended that Python virtual environments are
created using the copies of Python installed by `install_python()`. For
example:

    library(reticulate)
    version <- "3.9.12"
    install_python(version)
    virtualenv_create("my-environment", version = version)
    use_virtualenv("my-environment")

    # There is also support for a ":latest" suffix to select the latest patch release
    install_python("3.9:latest") # install latest patch available at python.org

    # select the latest 3.9.* patch installed locally
    virtualenv_create("my-environment", version = "3.9:latest")

## Note

On macOS and Linux this will build Python from sources, which may take a
few minutes. Installation will be faster if some build dependencies are
preinstalled. See
<https://github.com/pyenv/pyenv/wiki#suggested-build-environment> for
example commands you can run to pre-install system dependencies
(requires administrator privileges).

For example, on macOS you can pre-run:

    brew install openssl readline sqlite3 xz zlib tcl-tk@8 libb2

If `optimized = TRUE`, (the default) Python is build with:

    PYTHON_CONFIGURE_OPTS="--enable-shared --enable-optimizations --with-lto"
    PYTHON_CFLAGS="-march=native -mtune=native"

If `optimized = FALSE`, Python is built with:

    PYTHON_CONFIGURE_OPTS=--enable-shared

On Windows, prebuilt installers from <https://www.python.org> are used.
