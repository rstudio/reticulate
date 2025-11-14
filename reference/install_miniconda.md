# Install Miniconda

Download the [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
installer, and use it to install Miniconda.

## Usage

``` r
install_miniconda(path = miniconda_path(), update = TRUE, force = FALSE)
```

## Arguments

- path:

  The location where Miniconda is (or should be) installed. Note that
  the Miniconda installer does not support paths containing spaces. See
  [miniconda_path](https://rstudio.github.io/reticulate/reference/miniconda_path.md)
  for more details on the default path used by `reticulate`.

- update:

  Boolean; update to the latest version of Miniconda after installation?

- force:

  Boolean; force re-installation if Miniconda is already installed at
  the requested path?

## Details

For arm64 builds of R on macOS, `install_miniconda()` will use binaries
from [miniforge](https://github.com/conda-forge/miniforge) instead.

## Note

If you encounter binary incompatibilities between R and Miniconda, a
scripted build and installation of Python from sources can be performed
by
[`install_python()`](https://rstudio.github.io/reticulate/reference/install_python.md)

## See also

Other miniconda-tools:
[`miniconda_uninstall()`](https://rstudio.github.io/reticulate/reference/miniconda_uninstall.md),
[`miniconda_update()`](https://rstudio.github.io/reticulate/reference/miniconda_update.md)
