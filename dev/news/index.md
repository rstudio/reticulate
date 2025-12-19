# Changelog

## reticulate (development version)

- Fix spurious `Error: ignoring SIGPIPE signal` on Unix when embedded
  Python writes to a closed pipe
  ([\#1868](https://github.com/rstudio/reticulate/issues/1868)).

## reticulate 1.44.1

CRAN release: 2025-11-14

- The default Python version in
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  and
  [`conda_create()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  is now 3.12
  ([\#1862](https://github.com/rstudio/reticulate/issues/1862)).

- Fix error in
  [`virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)/[`virtualenv_starter()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  ([\#1861](https://github.com/rstudio/reticulate/issues/1861)).

- Fix display of Python variables in the Positron Variables Pane
  ([\#1859](https://github.com/rstudio/reticulate/issues/1859)).

## reticulate 1.44.0

CRAN release: 2025-10-25

- Reticulate now clears its cache automatically every 120 days.
  Configure the interval in `.Rprofile` with:
  `options(reticulate.max_cache_age = as.difftime(30, units = "days"))`.

- [`install_miniconda()`](https://rstudio.github.io/reticulate/dev/reference/install_miniconda.md)
  now installs miniforge instead of miniconda
  ([\#1800](https://github.com/rstudio/reticulate/issues/1800),
  [\#1820](https://github.com/rstudio/reticulate/issues/1820)).

- New helpers for reading and writing requirements files,
  [`py_write_requirements()`](https://rstudio.github.io/reticulate/dev/reference/py_requirements_files.md)
  and
  [`py_read_requirements()`](https://rstudio.github.io/reticulate/dev/reference/py_requirements_files.md)
  ([\#1846](https://github.com/rstudio/reticulate/issues/1846)).

- Adds support for requesting Python versions with a wildcard pattern
  `x.x.*` such as `==3.12.*` in
  [`virtualenv_starter()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md),
  [`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md),
  and related functions
  ([\#1825](https://github.com/rstudio/reticulate/issues/1825))

- Restored compatability with `uv` versions \>= 0.8.0
  ([\#1818](https://github.com/rstudio/reticulate/issues/1818)).

- [`virtualenv_starter()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  now discovers Python installations managed by external `uv`
  installations
  ([\#1849](https://github.com/rstudio/reticulate/issues/1849)).

- [`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md)
  now gives a better message when a user erroneously declares a module
  from the Python standard library as a required package
  ([@lazappi](https://github.com/lazappi),
  [\#1788](https://github.com/rstudio/reticulate/issues/1788))

- Positron’s reticulate integration will now be automatically enabled
  when the reticulate package is loaded in Positron
  ([\#1822](https://github.com/rstudio/reticulate/issues/1822)).

- [`with()`](https://rdrr.io/r/base/with.html) now forwards errors to
  Python context manager exit handlers (e.g., so database transactions
  can roll back cleanly)
  ([\#1840](https://github.com/rstudio/reticulate/issues/1840),
  [\#1841](https://github.com/rstudio/reticulate/issues/1841))

- Fixed `!!` in string literals being wrongly expanded to a `%system`
  magic in
  [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md).
  Added support for assigning `%system` command output to multiple
  variables via unpacking
  ([\#1844](https://github.com/rstudio/reticulate/issues/1844)).

- reticulate now warns when
  [`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md)d
  packages are not found in the selected Python virtual environment.
  This behavior can be disabled by setting the environment variable
  `RETICULATE_CHECK_REQUIRED_PACKAGES=0`
  ([\#1850](https://github.com/rstudio/reticulate/issues/1850)).

## reticulate 1.43.0

CRAN release: 2025-07-21

- Fixed usage of micromamba and mamba, next-generation conda environment
  management tools. reticulate now prefers to use micromamba, mamba,
  then conda when managing environments
  ([@gdevenyi](https://github.com/gdevenyi),
  [\#1771](https://github.com/rstudio/reticulate/issues/1771)).

- Added [`str()`](https://rdrr.io/r/utils/str.html),
  [`dim()`](https://rdrr.io/r/base/dim.html), and
  [`t()`](https://rdrr.io/r/base/t.html) S3 methods for NumPy Arrays.

- Fixed a segfault observed on R session exit
  ([\#1785](https://github.com/rstudio/reticulate/issues/1785),
  [\#1786](https://github.com/rstudio/reticulate/issues/1786)).

- Added check in
  [`install_miniconda()`](https://rstudio.github.io/reticulate/dev/reference/install_miniconda.md)
  if existing files will be overwritten.
  ([\#1794](https://github.com/rstudio/reticulate/issues/1794),
  [\#1796](https://github.com/rstudio/reticulate/issues/1796))

- Fixed error in
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  under R 4.5 when the requested Python version has a `":latest"`
  suffix, as it does by default.
  ([\#1792](https://github.com/rstudio/reticulate/issues/1792),
  [\#1797](https://github.com/rstudio/reticulate/issues/1797))

- Fixed error in `get_python_conda_info()` when conda not found through
  `conda-meta/history` and `NULL` is passed to `normalizePath`
  ([\#1184](https://github.com/rstudio/reticulate/issues/1184))

- Hotfix to pin `uv` version resolved by reticulate to `<0.8.0`.
  ([\#1812](https://github.com/rstudio/reticulate/issues/1812))

- Python discovery by `uv` is much faster now. The internal utility
  `uv_python_list()` searches only for managed python environments by
  default. Users can request discovery of system pythons by setting
  `UV_PYTHON_PREFERENCE`. Also, `uv_python_list()` will now discover
  pyenv pythons and python binaries installed by
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  if a system python is requested.
  ([\#1810](https://github.com/rstudio/reticulate/issues/1810))

## reticulate 1.42.0

CRAN release: 2025-03-25

- Fixed an issue in RStudio on Windows where interrupts were ignored
  while Python code was executing
  ([\#1753](https://github.com/rstudio/reticulate/issues/1753)).

- Updates for Positron to fix issues with
  [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md)
  and Variables Pane
  ([\#1755](https://github.com/rstudio/reticulate/issues/1755)).

- Fixed an issue where `[` received Python objects as slice arguments.
  e.g., `x[start:end]` when `start` or `end` were Python objects
  ([\#1731](https://github.com/rstudio/reticulate/issues/1731)).

- The `[` method will now translate symbol `..` to a Python Ellipsis
  `...`. ([\#1763](https://github.com/rstudio/reticulate/issues/1763))

- The `[` method can now accept index values greater than 2^31
  ([\#1769](https://github.com/rstudio/reticulate/issues/1769))

- Reticulate-managed `uv` can now resolve system-installed Pythons,
  supporting platforms where pre-built binaries are unavailable, such as
  musl-based Alpine Linux
  ([\#1751](https://github.com/rstudio/reticulate/issues/1751),
  [\#1752](https://github.com/rstudio/reticulate/issues/1752)).

- [`uv_run_tool()`](https://rstudio.github.io/reticulate/dev/reference/uv_run_tool.md)
  gains an `exclude_newer` argument
  ([\#1748](https://github.com/rstudio/reticulate/issues/1748)).

- [`py_register_load_hook()`](https://rstudio.github.io/reticulate/dev/reference/py_register_load_hook.md)
  is now exported to enable usage described in the “Using reticulate in
  an R package vignette”
  ([\#1754](https://github.com/rstudio/reticulate/issues/1754)).
  <https://rstudio.github.io/reticulate/articles/package.html>

- Internal changes to support R-devel (4.5) and R API updates
  ([\#1747](https://github.com/rstudio/reticulate/issues/1747),
  [\#1774](https://github.com/rstudio/reticulate/issues/1774)).

- Internal fixes to prevent reticulate-managed `uv` from writing outside
  reticulates cache directory
  ([\#1745](https://github.com/rstudio/reticulate/issues/1745)).

- Fixed an issue with pointing reticulate at a pyenv shim python
  ([\#1758](https://github.com/rstudio/reticulate/issues/1758))

## reticulate 1.41.0

CRAN release: 2025-02-24

- New
  [`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md)
  function for declaring Python requirements for the current R session.
  For details, see updated vignettes and help:

  - Installing Python Packages:
    <https://rstudio.github.io/reticulate/dev/articles/python_packages.html>
  - Using reticulate in an R Package:
    <https://rstudio.github.io/reticulate/dev/articles/package.html>
  - [`py_require()`](https://rstudio.github.io/reticulate/dev/reference/py_require.md)
    help:
    <https://rstudio.github.io/reticulate/dev/reference/py_require.html>

- New
  [`uv_run_tool()`](https://rstudio.github.io/reticulate/dev/reference/uv_run_tool.md)
  function for running command line tools distributed via Python
  packages.

- Raw R arrays and NumPy arrays with dtype “V1” (“void8”) now convert
  between each other. Use `r_to_py(as.array(x))` to efficiently convert
  raw vectors to NumPy arrays, and `py_to_r(array$view("V1"))` to
  efficiently convert NumPy arrays to raw vectors.
  ([\#1734](https://github.com/rstudio/reticulate/issues/1734))

- Fixed an issue with using Python 3.12 on Linux
  ([\#1712](https://github.com/rstudio/reticulate/issues/1712),
  [\#1714](https://github.com/rstudio/reticulate/issues/1714)).

- Fixed an issue where
  [`virtualenv_starter()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  would not discover a custom built Python
  ([\#1704](https://github.com/rstudio/reticulate/issues/1704)).

## reticulate 1.40.0

CRAN release: 2024-11-15

- The S3 classes for some (rarely encountered) Python objects have
  changed. Only Python objects with non-standard `__module__` values are
  affected. If a Python object’s parent class’s `__module__` attribute
  does not resolve to a string, reticulate:

  - Attempts to resolve it from the class’s class, if it’s a metaclass.
  - If no string can be resolved, reticulate no longer implicitly
    prepends ‘python.builtin.’ as the class prefix, instead it uses just
    the `__name__`. (See
    [\#1686](https://github.com/rstudio/reticulate/issues/1686) for more
    context)

- Added support for Python 3.13. Note that Python 3.13 removed support
  for `classmethod` descriptors, which may affect the S3 class of some
  Python objects that use metaclass properties to resolve a class’s
  `__module__` or `__name__` attribute.
  ([\#1686](https://github.com/rstudio/reticulate/issues/1686),
  [\#1698](https://github.com/rstudio/reticulate/issues/1698))

- [`py_is_null_xptr()`](https://rstudio.github.io/reticulate/dev/reference/py_is_null_xptr.md)
  and `[[` now load delayed modules
  ([\#1688](https://github.com/rstudio/reticulate/issues/1688)).

- Fixed error when attempting to use a python venv created with `uv`
  ([\#1678](https://github.com/rstudio/reticulate/issues/1678))

- Resolved an issue where
  [`py_discover_config()`](https://rstudio.github.io/reticulate/dev/reference/py_discover_config.md)
  attempted to detect Windows App Store Python installations. These are
  now excluded from discovery by both
  [`py_discover_config()`](https://rstudio.github.io/reticulate/dev/reference/py_discover_config.md)
  and
  [`virtualenv_starter()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  ([\#1656](https://github.com/rstudio/reticulate/issues/1656),
  [\#1673](https://github.com/rstudio/reticulate/issues/1673)).

- Fixed an error when converting an empty NumPy char array to R
  ([\#1662](https://github.com/rstudio/reticulate/issues/1662)).

- Fixed an error when using reticulate with radian
  ([\#1668](https://github.com/rstudio/reticulate/issues/1668),
  [\#1670](https://github.com/rstudio/reticulate/issues/1670)).

- Fixed a segfault encountered when running the Python session finalizer
  ([\#1663](https://github.com/rstudio/reticulate/issues/1663),
  [\#1664](https://github.com/rstudio/reticulate/issues/1664)).

- Resolved a segfault in RStudio when rapidly switching between R and
  Python chunks in a Quarto document
  ([\#1665](https://github.com/rstudio/reticulate/issues/1665)).

- Improved behavior when the conda binary used to create an environment
  cannot be resolved (contributed by
  [@tl-hbk](https://github.com/tl-hbk),
  [\#1654](https://github.com/rstudio/reticulate/issues/1654),
  [\#1659](https://github.com/rstudio/reticulate/issues/1659)).

- Added Positron support for the Variables Pane and
  [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md)
  ([\#1692](https://github.com/rstudio/reticulate/issues/1692),
  [\#1641](https://github.com/rstudio/reticulate/issues/1641),
  [\#1648](https://github.com/rstudio/reticulate/issues/1648),
  [\#1658](https://github.com/rstudio/reticulate/issues/1658),
  [\#1681](https://github.com/rstudio/reticulate/issues/1681),
  [\#1687](https://github.com/rstudio/reticulate/issues/1687)).

## reticulate 1.39.0

CRAN release: 2024-09-05

- Python background threads can now run in parallel with the R session
  ([\#1641](https://github.com/rstudio/reticulate/issues/1641)).

- [`py_main_thread_func()`](https://rstudio.github.io/reticulate/dev/reference/py_main_thread_func.md)
  is deprecated; every R function can now safely be called from
  background Python threads
  ([\#1648](https://github.com/rstudio/reticulate/issues/1648)).

- Calls from Python threads into R now notify the main thread using R’s
  native event loop, ensuring that these calls are handled even when the
  main thread is engaged in non-Python tasks
  ([\#1648](https://github.com/rstudio/reticulate/issues/1648)).

- The knitr engine now avoids overwriting Altair’s default chart
  dimensions with the values of `ut.width.px` and `ut.height.px`. Use
  `altair.fig.height`, `altair.fig.width`, or Altair’s `width` and
  `height` parameters to adjust chart dimensions (contributed by
  [@joelostblom](https://github.com/joelostblom),
  [\#1646](https://github.com/rstudio/reticulate/issues/1646)).

- New [`as.character()`](https://rdrr.io/r/base/character.html) method
  for `python.builtin.str` with support for handling embedded NULs in
  strings ([\#1653](https://github.com/rstudio/reticulate/issues/1653)).

- New [`as.raw()`](https://rdrr.io/r/base/raw.html) method for
  `python.builtin.bytes`
  ([\#1649](https://github.com/rstudio/reticulate/issues/1649),
  [\#1652](https://github.com/rstudio/reticulate/issues/1652)).

- [`as.character()`](https://rdrr.io/r/base/character.html) method for
  `python.builtin.bytes` gains a `nul` argument, allowing for convenient
  handling of embedded NULs in the string
  ([\#1652](https://github.com/rstudio/reticulate/issues/1652)).

- Reticulate now uses the `RETICULATE_VIRTUALENV_ROOT` environment
  variable when determining where to resolve virtual environments
  ([\#1657](https://github.com/rstudio/reticulate/issues/1657)).

- [`conda_run2()`](https://rstudio.github.io/reticulate/dev/reference/conda_run2.md)
  is now exported (contributed by
  [@dramanica](https://github.com/dramanica),
  [\#1637](https://github.com/rstudio/reticulate/issues/1637)).

- The Python session is now finalized when the R session exits
  ([\#1648](https://github.com/rstudio/reticulate/issues/1648)).

- Internal updates for NumPy 2.1
  ([\#1651](https://github.com/rstudio/reticulate/issues/1651)).

- Fixed error when importing a module named `config`
  ([\#1628](https://github.com/rstudio/reticulate/issues/1628)).

- Fixes for CRAN check failures on macOS-oldrel
  ([\#1645](https://github.com/rstudio/reticulate/issues/1645)).

- Fixed an error where opening a Python subprocess in Positron on
  Windows resulted in “OSError: \[WinError 6\] The handle is invalid.”
  ([\#1658](https://github.com/rstudio/reticulate/issues/1658),
  posit-dev/positron#4457).

## reticulate 1.38.0

CRAN release: 2024-06-19

- Python Exceptions converted to R conditions are now R lists instead of
  R environments, for compatability with {rlang} and {purrr}.
  (tidyverse/purrr#1104, r-lib/rlang#1664,
  [\#1617](https://github.com/rstudio/reticulate/issues/1617))

- Internal updates for NumPy 2.0
  ([\#1621](https://github.com/rstudio/reticulate/issues/1621))

- Added support for converting NumPy StringDType arrays to R character
  arrays. ([\#1623](https://github.com/rstudio/reticulate/issues/1623))

- Internal updates for compliance with R’s upcoming formalized C API.
  ([\#1625](https://github.com/rstudio/reticulate/issues/1625))

- Fixed an issue where attempting to convert a NumPy array with a
  non-simple dtype to R would signal an error.
  ([\#1613](https://github.com/rstudio/reticulate/issues/1613), fixed in
  [\#1614](https://github.com/rstudio/reticulate/issues/1614)).

## reticulate 1.37.0

CRAN release: 2024-05-21

- Interrupting Python no longer leads to segfaults.
  ([\#1601](https://github.com/rstudio/reticulate/issues/1601), fixed in
  [\#1602](https://github.com/rstudio/reticulate/issues/1602))

- Print method for Python callables now includes the callable’s
  signature.
  ([\#1605](https://github.com/rstudio/reticulate/issues/1605),
  [\#1607](https://github.com/rstudio/reticulate/issues/1607))

- Reticulate now installs successfully on Windows ARM64.
  ([\#1609](https://github.com/rstudio/reticulate/issues/1609),
  contributed by [@andrjohns](https://github.com/andrjohns))

- [`virtualenv_starter()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  no longer warns when encountering broken symlinks.
  ([\#1598](https://github.com/rstudio/reticulate/issues/1598))

- Fixed an issue where configuration for reticulate `conda_*` functions
  to use the executable `mamba` instead of `conda` was ignored.
  ([\#1608](https://github.com/rstudio/reticulate/issues/1608),
  contributed by
  [@AlexandreGuinaudeau](https://github.com/AlexandreGuinaudeau))

## reticulate 1.36.1

CRAN release: 2024-04-22

- Fix issue where
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  method for Pandas DataFrames would error if
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  S3 methods were defined for Pandas subtypes, (as done by {anndata})
  ([\#1591](https://github.com/rstudio/reticulate/issues/1591)).

- “Python Dependencies” vignette edits
  ([@salim-b](https://github.com/salim-b),
  [\#1586](https://github.com/rstudio/reticulate/issues/1586))

- Added an option for extra command-line arguments in
  [`conda_create()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  and
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  ([\#1585](https://github.com/rstudio/reticulate/issues/1585)).

- Fixed issue where
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  would ignore user-specified channels during Python installation
  ([\#1594](https://github.com/rstudio/reticulate/issues/1594)).

## reticulate 1.36.0

CRAN release: 2024-04-15

- Internal refactoring and optimizations now give a faster experience,
  especially for workflows that frequently access Python objects from R.
  For example, simple attribute access like `sys$path` is ~2.5x times
  faster, and a sample workload of
  `py_to_r(np_array(1:3) + np_array(1:3))` benchmarks ~3.5x faster when
  compared to the previous CRAN release.

- Fixed issue where callable python objects created with
  `convert = FALSE` would not be wrapped in an R function
  ([\#1522](https://github.com/rstudio/reticulate/issues/1522)).

- Fixed issue where
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  S3 methods would not be called on arguments supplied to R functions
  being called from Python
  ([\#1522](https://github.com/rstudio/reticulate/issues/1522)).

- [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  will now build optimized versions of Python on macOS and Linux
  ([\#1567](https://github.com/rstudio/reticulate/issues/1567))

- Default Python version installed by
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  is now 3.10 (was 3.9)
  ([\#1574](https://github.com/rstudio/reticulate/issues/1574)).

- Output of
  [`reticulate::py_last_error()`](https://rstudio.github.io/reticulate/dev/reference/py_last_error.md)
  now includes a hint, showing how to access the full R call stack
  ([\#1572](https://github.com/rstudio/reticulate/issues/1572)).

- Fixed an issue where nested
  [`py_capture_output()`](https://rstudio.github.io/reticulate/dev/reference/py_capture_output.md)
  calls result in a lost reference to the original `sys.stdout` and
  `sys.stderr`, resulting in no further visible output from Python, and
  eventually, a segfault.
  ([\#1564](https://github.com/rstudio/reticulate/issues/1564))

- Fixed issues reported by rchk, as requested by CRAN
  ([\#1581](https://github.com/rstudio/reticulate/issues/1581)).

- `py_to_r(x)` now returns `x` unmodified if `x` is not a Python object,
  instead of signaling an error.

- New [`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html)
  method exported for Python Polars DataFrames
  ([\#1568](https://github.com/rstudio/reticulate/issues/1568))

- Fixed an issue where printing a delayed module
  (`import("foo", delay_load = TRUE)`) would output `<pointer: 0x0>`.

- [`py_validate_xptr()`](https://rstudio.github.io/reticulate/dev/reference/py_is_null_xptr.md)
  will now attempt to resolve delayed modules before signaling an error
  ([\#1561](https://github.com/rstudio/reticulate/issues/1561)).

- R packages can now express multiple preferred Python environments to
  search for and use if they exist, by supplying a character vector to
  [`import()`](https://rstudio.github.io/reticulate/dev/reference/import.md):
  `import("foo", delay_load = list(environment = c("r-foo", "r-bar")))`
  ([\#1559](https://github.com/rstudio/reticulate/issues/1559))

- Reticulate will no longer warn about ignored
  `use_python(,required = FALSE)` calls
  ([\#1562](https://github.com/rstudio/reticulate/issues/1562)).

- `reticulate` now prefers using the agg matplotlib backend when the R
  session is non-interactive. The backend can also be overridden via the
  `MPLBACKEND` or `RETICULATE_MPLBACKEND` environment variables when
  necessary
  ([\#1556](https://github.com/rstudio/reticulate/issues/1556)).

- `attr(x, "tzone")` attributes are (better) preserved when converting
  POSIXt to Python. POSIXt types with a non-empty `tzone` attr convert
  to a `datetime.datetime`, otherwise they convert to NumPy
  `datetime64[ns]` arrays.

- Fixed an issue where calling
  [`py_set_item()`](https://rstudio.github.io/reticulate/dev/reference/py_get_item.md)
  on a subclassed dict would not invoke a custom `__setitem__` method.

- `py_del_attr(x, name)` now returns `x` invisibly

- [`source_python()`](https://rstudio.github.io/reticulate/dev/reference/source_python.md)
  no longer exports the `r` symbol to the R global environment. (the “R
  Interface Object” that is used by Python code get a reference to the R
  [`globalenv()`](https://rdrr.io/r/base/environment.html))

- Fixed hang encountered (sometimes) when attempting to call
  [`iterate()`](https://rstudio.github.io/reticulate/dev/reference/iterate.md)
  on an exhausted
  [`py_iterator()`](https://rstudio.github.io/reticulate/dev/reference/py_iterator.md)
  object multiple times
  ([\#1539](https://github.com/rstudio/reticulate/issues/1539)).

- `iterate(simplify=TRUE)` rewritten in C for speed improvements
  ([\#1539](https://github.com/rstudio/reticulate/issues/1539)).

- Update for Pandas 2.2 deprecation of `Index.format()`
  ([\#1537](https://github.com/rstudio/reticulate/issues/1537),
  [\#1538](https://github.com/rstudio/reticulate/issues/1538)).

- Updates for CRAN R-devel (R 4.4)
  ([\#1554](https://github.com/rstudio/reticulate/issues/1554)).

- Fixed an issue where
  [`py_discover_config()`](https://rstudio.github.io/reticulate/dev/reference/py_discover_config.md)
  would discover `python` (v2) on the PATH in preference of `python3` on
  the PATH.
  ([\#1547](https://github.com/rstudio/reticulate/issues/1547))

- Fixed an issue where reticulate would error when using conda
  environments created with the (new) `conda env create` command.
  ([\#1535](https://github.com/rstudio/reticulate/issues/1535),
  [\#1543](https://github.com/rstudio/reticulate/issues/1543))

- Fixed an issue where reticulate would error when using a conda
  environment where the original conda binary that was used to create
  the environment is no longer available
  ([\#1555](https://github.com/rstudio/reticulate/issues/1555))

- Fixed an issue where a user would be unable to accept the prompt to
  create the default “r-reticulate” venv
  ([\#1557](https://github.com/rstudio/reticulate/issues/1557)).

- [`is_py_object()`](https://rstudio.github.io/reticulate/dev/reference/is_py_object.md)
  is now exported
  ([\#1573](https://github.com/rstudio/reticulate/issues/1573)).

## reticulate 1.35.0

CRAN release: 2024-01-31

- Subclassed Python list and dict objects are no longer automatically
  converted to R vectors. Additionally, the S3 R `class` attribute for
  Python objects is now constructed using the Python `type(object)`
  directly, rather than from the `object.__class__` attribute. See
  [\#1531](https://github.com/rstudio/reticulate/issues/1531) for
  details and context.

- R external pointers (EXTPTRSXP objects) now round-trip through
  `py_to_r(r_to_py(x))` successfully. (reported in
  [\#1511](https://github.com/rstudio/reticulate/issues/1511), fixed in
  [\#1519](https://github.com/rstudio/reticulate/issues/1519),
  contributed by [@llaniewski](https://github.com/llaniewski)).

- Fixed issue where
  [`virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  would error on Ubuntu 22.04 when using the system python as a base.
  ([\#1495](https://github.com/rstudio/reticulate/issues/1495), fixed in
  [\#1496](https://github.com/rstudio/reticulate/issues/1496)).

- Fixed issue where `csc_matrix` objects with unsorted indices could not
  be converted to a dgCMatrix. (related to
  [\#727](https://github.com/rstudio/reticulate/issues/727), fixed in
  [\#1524](https://github.com/rstudio/reticulate/issues/1524),
  contributed by [@rcannood](https://github.com/rcannood)).

- Added support for partially unexpanded variables like `$USER` in
  `XDG_DATA_HOME` and similar
  ([\#1513](https://github.com/rstudio/reticulate/issues/1513),
  [\#1514](https://github.com/rstudio/reticulate/issues/1514))

### Knitr Python Engine Changes:

- The knitr python engine now formats captured python exceptions to
  include the exception type and any exception notes when chunk options
  `error = TRUE` is set (reported in
  [\#1520](https://github.com/rstudio/reticulate/issues/1520), fixed in
  [\#1527](https://github.com/rstudio/reticulate/issues/1527)).

- Fixed an issue where the knitr python engine would fail to include
  figures from python chunks if a custom `root.dir` chunk option was
  set. (reported in
  [\#1526](https://github.com/rstudio/reticulate/issues/1526), fixed in
  [\#1529](https://github.com/rstudio/reticulate/issues/1529))

- knitr engine gains the ability to save chunk figures in multiple
  files/formats (Contributed by [@Rumengol](https://github.com/Rumengol)
  in [\#1507](https://github.com/rstudio/reticulate/issues/1507))

- Fixed an issue where matplotlib figures generated in the initial chunk
  where matplotlib was first imported would be the wrong size (reported
  in [\#1523](https://github.com/rstudio/reticulate/issues/1523), fixed
  in [\#1530](https://github.com/rstudio/reticulate/issues/1530))

- Fixed an issue where the knitr engine would not correctly display
  altair compound charts if more than one were present in a document
  ([\#1500](https://github.com/rstudio/reticulate/issues/1500),
  [\#1532](https://github.com/rstudio/reticulate/issues/1532)).

## reticulate 1.34.0

CRAN release: 2023-10-12

## reticulate 1.33.0

- Fixed issue where `asyncio`, (and modules that use `asyncio`), would
  error on Windows when running under RStudio
  ([\#1478](https://github.com/rstudio/reticulate/issues/1478),
  [\#1479](https://github.com/rstudio/reticulate/issues/1479)).

- Added compatability with Python 3.12.

- [`condaenv_exists()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  is now exported.

## reticulate 1.32.0

CRAN release: 2023-09-11

- reticulate now supports casting R data.frames to Pandas data.frames
  using nullable data types, allowing users to preserve NA’s from R
  atomic vectors. This feature is opt-in and can be enabled by setting
  the R option `reticulate.pandas_use_nullable_dtypes` to `TRUE`.
  ([\#1439](https://github.com/rstudio/reticulate/issues/1439))

- reticulate now exports a
  [`chooseOpsMethod()`](https://rdrr.io/r/base/chooseOpsMethod.html)
  method, allowing for Ops dispatch to more specialized Ops methods
  defined for Python objects.

- [`py_discover_config()`](https://rstudio.github.io/reticulate/dev/reference/py_discover_config.md)
  will now warn instead of error upon encountering a broken Python
  installation.
  ([\#1441](https://github.com/rstudio/reticulate/issues/1441),
  [\#1459](https://github.com/rstudio/reticulate/issues/1459))

- Fixed issue where Python would raise exception “OSError: \[WinError
  6\] The handle is invalid” when opening a subprocess while running in
  Rstudio on Windows.
  ([\#1448](https://github.com/rstudio/reticulate/issues/1448),
  [\#518](https://github.com/rstudio/reticulate/issues/518))

- Fixed issue where the multiprocessing Python module would crash or
  hang when spawning a `Process()` on Windows.
  ([\#1430](https://github.com/rstudio/reticulate/issues/1430),
  [\#1346](https://github.com/rstudio/reticulate/issues/1346), fixed in
  [\#1461](https://github.com/rstudio/reticulate/issues/1461))

- Fixed issue where
  [`virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  would fail to discover a ‘virtualenv’ module in the system Python
  installation on Ubuntu. Reticulate will no longer discover and attempt
  to use the `venv` module stub present on Ubuntu systems where the
  `python3-venv` apt package has not been installed.
  (mlverse/pysparklyr#11,
  [\#1437](https://github.com/rstudio/reticulate/issues/1437),
  [\#1455](https://github.com/rstudio/reticulate/issues/1455))

- Fixed issue where the user was prompted to create an ‘r-reticulate’
  venv in the RStudio IDE before reticulate was requested to initialize
  Python. ([\#1450](https://github.com/rstudio/reticulate/issues/1450),
  [\#1456](https://github.com/rstudio/reticulate/issues/1456))

- Improved error message when reticulate attempts to initialize a
  virtual environment after the Python installation it was created from
  is no longer available.
  ([\#1149](https://github.com/rstudio/reticulate/issues/1149),
  [\#1457](https://github.com/rstudio/reticulate/issues/1457))

- Improved error message on Fedora when attempting to create a virtual
  environment from the system python before running
  `dnf install python3-pip`.

- Fixed issue where
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  on macOS in the RStudio IDE would fail to discover and use brew for
  Python build dependencies.

- Fixed error with `virtualenv_create(python = "/usr/bin/python")` on
  centos7. ([\#1467](https://github.com/rstudio/reticulate/issues/1467))

## reticulate 1.31

CRAN release: 2023-08-10

### Python Installation Management

- reticulate will no longer prompt users to install miniconda. Instead,
  reticulate will now prompt users to create a default `r-reticulate`
  venv.

- The search that reticulate conducts to select which Python
  installation to load has changed. See the updated Python “Order of
  Discover” in the “versions” vignette.
  [`vignette("versions", package = "reticulate")`](https://rstudio.github.io/reticulate/dev/articles/versions.md).

- Updated recommendations in the “python_dependencies” vignette for how
  R packages can approach Python dependency management.
  [`vignette("python_dependencies", package = "reticulate")`](https://rstudio.github.io/reticulate/dev/articles/python_dependencies.md)

- New function
  [`virtualenv_starter()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md),
  which can be used to find a suitable python binary for creating a
  virtual environment. This is now the default method for finding the
  python binary when calling `virtualenv_create(version = <version>)`.

- [`virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  and
  [`virtualenv_install()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  gain a `requirements` argument, accepting a filepath to a python
  requirements file.

- [`virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  gains a `force` argument.

- [`virtualenv_install()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  gains a `python_version` argument, allowing users to customize which
  python version is used when bootstrapping a new virtual environment.

- Fixed an issue where the list of available python versions used by
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  would be out-of-date.

- [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  now gives a better error message if git is not installed.

- [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  on macOS will now will use brew, if it’s available, to install build
  dependencies, substantially speeding up python build times.

- New function
  [`conda_search()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md),
  contributed by [@mkoohafkan](https://github.com/mkoohafkan) in PR
  [\#1364](https://github.com/rstudio/reticulate/issues/1364).

### Language

- New `[` and `[<-` methods that invoke Python `__getitem__`,
  `__setitem__` and `__delitem__`. The R generics `[` and `[<-` now
  accept python-style slice syntax like `x[1:2:3]`. See examples in
  [`?py_get_item`](https://rstudio.github.io/reticulate/dev/reference/py_get_item.md).

- [`py_iterator()`](https://rstudio.github.io/reticulate/dev/reference/py_iterator.md)
  gains a `prefetch` argument, primarily to avoid deadlocks where the
  main thread is blocked, waiting for the iterator, which is waiting to
  run on the main thread, as encountered in TensorFlow/Keras.
  ([\#1405](https://github.com/rstudio/reticulate/issues/1405)).

- String columns from Pandas data frames containing `None`, `pd.NA` or
  `np.nan` are now simplified into character vectors and missing values
  replaced by `NA`
  ([\#1428](https://github.com/rstudio/reticulate/issues/1428)).

- Converting from Pandas data frames containing columns with Pandas
  nullable data types are now correctly converted into R data.frames
  preserving the missing values
  ([\#1427](https://github.com/rstudio/reticulate/issues/1427)).

### Knitr

- The knitr engine gains a `jupyter_compat` option, enabling reticulate
  to better match the behavior of Jupyter. When this chunk option is set
  to `TRUE`, only the return value from the last expression in a chunk
  is auto-printed.
  ([\#1391](https://github.com/rstudio/reticulate/issues/1391),
  [\#1394](https://github.com/rstudio/reticulate/issues/1394),
  contributed by [@matthew-brett](https://github.com/matthew-brett))

- The knitr engine now more reliably detects and displays matplotlib
  pending plots, without the need for a matplotlib artist object to be
  returned as a top-level expression. E.g., the knitr engine will now
  display plots when the matplotlib api returns something other than an
  artist object, (`plt.bar()`), or the matplotlib return value is not
  auto-printed due to being assigned, (`x = plt.plot()`), or suppressed
  with a `;`, (`plt.plot();`).
  ([\#1391](https://github.com/rstudio/reticulate/issues/1391),
  [\#1401](https://github.com/rstudio/reticulate/issues/1401),
  contributed by [@matthew-brett](https://github.com/matthew-brett))

- Fixed an issue where knitr engine would not respect chunk options
  `fig.width` / `fig.height` when rendering matplotlib plots.
  ([\#1398](https://github.com/rstudio/reticulate/issues/1398))

- Fixed an issue where the reticulate knitr engine would not capture
  output printed from python. (PR
  [\#1412](https://github.com/rstudio/reticulate/issues/1412), fixing
  [\#1378](https://github.com/rstudio/reticulate/issues/1378),
  [\#331](https://github.com/rstudio/reticulate/issues/331))

### Miscellanous

- Reticulate now periodically flushes python `stdout` and `stderr`
  buffers even while the main thread is blocked executing Python code.
  Streaming output from a long-running Python function call will now
  appear in the R console while the Python function is still executing.
  (Previously, output might not appear until the Python function had
  finished and control of the main thread had returned to R).

- Updated sparse matrix conversion routines for compatibility with scipy
  1.11.0.

- Fixed an issue where a py capsule finalizer could access the R API
  from a background thread.
  ([\#1406](https://github.com/rstudio/reticulate/issues/1406))

- Fixed issue where R would segfault (crash) in long-lived R sessions
  where both rpy2 and reticulate were in use
  ([\#1236](https://github.com/rstudio/reticulate/issues/1236)).

- Fixed an issue where exceptions from reticulate would not be formatted
  properly when running tests under testthat (r-lib/rlang#1637,
  [\#1413](https://github.com/rstudio/reticulate/issues/1413)).

- Fixed an issue where `py_get_attr(silent = TRUE)` would not return an
  R `NULL`, if the attribute was missing, as documented.
  ([\#1413](https://github.com/rstudio/reticulate/issues/1413))

- Fixed an issue where `py_get_attr(silent = TRUE)` would leave a python
  global exception set if the attribute was missing, resulting in fatal
  errors when running python under debug mode.
  ([\#1396](https://github.com/rstudio/reticulate/issues/1396))

## reticulate 1.30

CRAN release: 2023-06-09

- Fix compilation error on R 3.5. Bump minimum R version dependency to
  3.5.

## reticulate 1.29

CRAN release: 2023-06-05

#### Exceptions and Errors:

- R error information (call, message, other attributes) is now preserved
  as an R error condition traverses the R \<-\> Python boundary.

- Python Exceptions now inherit from `error` and `condition`, and can be
  passed directly to [`base::stop()`](https://rdrr.io/r/base/stop.html)
  to signal an error in R and raise an exception in Python.

- Raised Python Exceptions are now used directly to signal an R error.
  For example, in the following code, `e` is now an object that inherits
  from `python.builtin.Exception` as well as `error` and `condition`:
  `r e <- tryCatch(py_func_that_raises_exception(), error = function(e) e)`
  Use [`base::conditionCall()`](https://rdrr.io/r/base/conditions.html)
  and
  [`base::conditionMessage()`](https://rdrr.io/r/base/conditions.html)
  to access the original R call and error message.

- [`py_last_error()`](https://rstudio.github.io/reticulate/dev/reference/py_last_error.md)
  return object contains `r_call`, `r_trace` and/or `r_class` if the
  Python Exception was raised by an R function called from Python.

- The hint to run
  [`reticulate::py_last_error()`](https://rstudio.github.io/reticulate/dev/reference/py_last_error.md)
  after an exception is now clickable in the RStudio IDE.

- Filepaths to Python files in the print output from
  [`py_last_error()`](https://rstudio.github.io/reticulate/dev/reference/py_last_error.md)
  are now clickable links in the RStudio IDE.

- Python exceptions encountered in
  [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md)
  are now printed with the full Python traceback by default. In the
  RStudio IDE, filepaths in the tracebacks are rendered as clickable
  links. ([\#1240](https://github.com/rstudio/reticulate/issues/1240))

#### Language:

- Converted Python callables gain support for dynamic dots from the
  rlang package. New features:

  - splicing (unpacking) arguments: `fn(!!!kwargs)`
  - dynamic names: `nm <- "key"; fn("{nm}" := value)`
  - trailing commas ignored (matching Python syntax): `fn(a, )`
    identical to `fn(a)`

- New Ops group generics for Python objects: `+`, `-`, `*`, `/`, `^`,
  `%%`, `%/%`, `&`, `|`, `!`, `%*%`. Methods for all the Ops group
  generics are now defined for Python objects.
  ([\#1187](https://github.com/rstudio/reticulate/issues/1187),
  [\#1363](https://github.com/rstudio/reticulate/issues/1363)) E.g.,
  this now works:

  ``` r
  np <- reticulate::import("numpy", convert = FALSE)
  x <- np$array(1:5)
  y <- np$array(6:10)
  x + y
  ```

- Fixed two issues with R comparison operator methods (`==`, `!=`, `<`,
  `<=`, `>=`, `>`):

  - The operators no longer error on Python objects that define “rich
    comparison” Python methods that don’t return a single bool. (e.g.,
    numpy arrays).
  - The operators now respect the ‘convert’ value of the supplied Python
    objects. Note, this may be a breaking change as, e.g, `==`, may now
    no long return an R scalar logical if one of the Python object being
    compared was created with `convert = FALSE`. Wrap the result of the
    comparison with
    [`py_bool()`](https://rstudio.github.io/reticulate/dev/reference/py_bool.md)
    to restore the previous behavior.
    ([\#1187](https://github.com/rstudio/reticulate/issues/1187),
    [\#1363](https://github.com/rstudio/reticulate/issues/1363))

- R functions wrapping Python callables now have formals matching those
  of the Python callable signature, enabling better autocompletion in
  more contexts
  ([\#1361](https://github.com/rstudio/reticulate/issues/1361)).

- new [`nameOfClass()`](https://rdrr.io/r/base/class.html) S3 method for
  Python types, enabling usage:
  `base::inherits(x, <python-type-object>)` (requires R \>= 4.3.0)

- [`py_run_file()`](https://rstudio.github.io/reticulate/dev/reference/py_run.md)
  and
  [`source_python()`](https://rstudio.github.io/reticulate/dev/reference/source_python.md)
  now prepend the script directory to the Python module search path,
  `sys.path`, while the requested script is executing. This allows the
  Python scripts to resolve imports of modules defined in the script
  directory, matching the behavior of `python <script>` at the command
  line. ([\#1347](https://github.com/rstudio/reticulate/issues/1347))

#### knitr:

- The knitr engine now suppresses warnings from Python code if
  `warning=FALSE` is set in the chunk options. (quarto-dev/quarto#125,
  [\#1358](https://github.com/rstudio/reticulate/issues/1358))

- Fixed issue where reticulate’s knitr engine would attach comments in a
  code chunk to the wrong code chunk (requires Python\>=3.8)
  ([\#1223](https://github.com/rstudio/reticulate/issues/1223)).

- The knitr Python engine now respects the `strip.white` option
  ([\#1273](https://github.com/rstudio/reticulate/issues/1273)).

- Fixed issue where the knitr engine would show an additional plot from
  a chunk if the user called `matplotlib.pyplot.show()`
  ([\#1380](https://github.com/rstudio/reticulate/issues/1380),
  [\#1383](https://github.com/rstudio/reticulate/issues/1383))

#### Misc:

- [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  now succeeds when converting subtypes of the built-in types
  (e.g. `list`, `dict`, `str`).
  ([\#1352](https://github.com/rstudio/reticulate/issues/1352),
  [\#1348](https://github.com/rstudio/reticulate/issues/1348),
  [\#1226](https://github.com/rstudio/reticulate/issues/1226),
  [\#1354](https://github.com/rstudio/reticulate/issues/1354),
  [\#1366](https://github.com/rstudio/reticulate/issues/1366))

- New
  [`pillar::type_sum()`](https://pillar.r-lib.org/reference/type_sum.html)
  method now exported for Python objects. That ensures the full object
  class name is printing in R tracebacks and tibbles containing Python
  objects.

- [`py_load_object()`](https://rstudio.github.io/reticulate/dev/reference/py_save_object.md)
  gains a `convert` argument. If `convert = FALSE`, the returned Python
  object will not be converted to an R object.

- Fixed error
  [`r_to_py()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  with Pandas\>=2.0 and R data.frames with a factor column containing
  levels with `NA`.

- [`r_to_py()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  now succeeds for many additional types of R objects. Objects that
  reticulate doesn’t know how to convert are presented to the Python
  runtime as a pycapsule (an opaque pointer to the underlying R object).
  Previously this would error. This allows for R code to pass R objects
  that cannot be safely converted to Python through the Python runtime
  to other R code. (e.g, to an R function called by Python code).
  ([\#1304](https://github.com/rstudio/reticulate/issues/1304))

- reticulate gains the ability to bind to micromamba Python
  installations
  ([\#1378](https://github.com/rstudio/reticulate/issues/1378),
  [\#1176](https://github.com/rstudio/reticulate/issues/1176),
  [\#1382](https://github.com/rstudio/reticulate/issues/1382),
  [\#1379](https://github.com/rstudio/reticulate/issues/1379), thanks to
  Zia Khan, [@zia1138](https://github.com/zia1138))

- Default Python version used by
  [`install_miniconda()`](https://rstudio.github.io/reticulate/dev/reference/install_miniconda.md)
  and friends is now 3.9 (was 3.8).

## reticulate 1.28

CRAN release: 2023-01-27

- Fixed issue where
  [`source_python()`](https://rstudio.github.io/reticulate/dev/reference/source_python.md)
  (and likely many other entrypoints) would error if reticulate was
  built with Rcpp 1.0.10. Exception and error handling has been updated
  to accommodate usage of `R_ProtectUnwind()`.
  ([\#1328](https://github.com/rstudio/reticulate/issues/1328),
  [\#1329](https://github.com/rstudio/reticulate/issues/1329)).

- Fixed issue where reticulate failed to discover Python 3.11 on
  Windows. ([\#1325](https://github.com/rstudio/reticulate/issues/1325))

- Fixed issue where reticulate would error by attempting to bind to a
  cygwin/msys2 installation of Python on Windows
  ([\#1325](https://github.com/rstudio/reticulate/issues/1325)).

## reticulate 1.27

CRAN release: 2023-01-07

- [`py_run_file()`](https://rstudio.github.io/reticulate/dev/reference/py_run.md)
  now ensures the `__file__` dunder is visible to the executing python
  code. ([\#1283](https://github.com/rstudio/reticulate/issues/1283),
  [\#1284](https://github.com/rstudio/reticulate/issues/1284))

- Fixed errors with
  [`install_miniconda()`](https://rstudio.github.io/reticulate/dev/reference/install_miniconda.md)
  and
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md),
  on Windows
  ([\#1286](https://github.com/rstudio/reticulate/issues/1286),
  [\#1287](https://github.com/rstudio/reticulate/issues/1287),
  conda/conda#11795,
  [\#1312](https://github.com/rstudio/reticulate/issues/1312),
  [\#1297](https://github.com/rstudio/reticulate/issues/1297)), and on
  Linux and macOS
  ([\#1306](https://github.com/rstudio/reticulate/issues/1306),
  conda/conda#10431)

- Fixed error when activating a conda env from a UNC drive on Windows
  ([\#1303](https://github.com/rstudio/reticulate/issues/1303)).

## reticulate 1.26

CRAN release: 2022-08-31

- Fixed issue where reticulate failed to bind to python2.
  ([\#1241](https://github.com/rstudio/reticulate/issues/1241),
  [\#1229](https://github.com/rstudio/reticulate/issues/1229))

- A warning is now issued when reticulate binds to python2 that python2
  support will be removed in an upcoming reticulate release.

- [`py_id()`](https://rstudio.github.io/reticulate/dev/reference/py_id.md)
  now returns a character string, instead of an R integer
  ([\#1216](https://github.com/rstudio/reticulate/issues/1216)).

- Fixed an issue where
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  would not convert elements of a dictionary
  ([\#1221](https://github.com/rstudio/reticulate/issues/1221)).

- Fixed an issue where setting `RETICULATE_PYTHON` or
  `RETICULATE_PYTHON_FALLBACK` on Windows to the pyenv-win `python.bat`
  shim would result in an error
  ([\#1263](https://github.com/rstudio/reticulate/issues/1263)).

- Fixed an issue where `datetime.datetime` objects with a `tzinfo`
  attribute was not getting converted to R correctly
  ([\#1266](https://github.com/rstudio/reticulate/issues/1266)).

- Fixed an issue where pandas `pandas.Categorical(,ordered=True)` Series
  were not correctly converted to an R ordered factor
  ([\#1234](https://github.com/rstudio/reticulate/issues/1234)).

- The `reticulate` Python engine no longer halts on error for Python
  chunks containing parse errors when the `error=TRUE` chunk option is
  set. ([\#583](https://github.com/rstudio/reticulate/issues/583))

- [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  now leverages brew for python build dependencies like <openssl@1.1> if
  brew is already installed and on the PATH, substantially speeding up
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  on macOS systems with brew configured.

- Fixed an issue where reticulate would fail to bind to a conda
  environment on macOS or linux if conda installed a non-POSIX compliant
  activation script into the conda environment.
  ([\#1255](https://github.com/rstudio/reticulate/issues/1255))

- Fixed an issue where the python knitr engine would error when printing
  to HTML a constructor of class instances with a `_repr_html_` or
  `to_html` method (e.g., `pandas.DataFrame`;
  [\#1249](https://github.com/rstudio/reticulate/issues/1249),
  [\#1250](https://github.com/rstudio/reticulate/issues/1250)).

- Fixed an issue where the python knitr engine would error when printing
  a plotly figure to an HTML document in some (head-less) linux
  environments
  ([\#1250](https://github.com/rstudio/reticulate/issues/1250)).

- Fixed an issue where `conda_install(pip=TRUE)` would install packages
  into a user Python library instead of the conda env if the environment
  variable `PIP_USER=true` was set.
  [`py_install()`](https://rstudio.github.io/reticulate/dev/reference/py_install.md),
  [`virtualenv_install()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md),
  and
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  now always specify `--no-user` when invoking `pip install`.
  ([\#1209](https://github.com/rstudio/reticulate/issues/1209))

- Fixed issue where
  [`py_last_error()`](https://rstudio.github.io/reticulate/dev/reference/py_last_error.md)
  would return unconverted Python objects
  ([\#1233](https://github.com/rstudio/reticulate/issues/1233))

- The Knitr engine now supports printing Python objects with
  `_repr_markdown_` methods. (via quarto-dev/quarto-cli#1501)

- `sys.executable` on Windows now correctly reports the path to the
  Python executable instead of the launching R executable.
  ([\#1258](https://github.com/rstudio/reticulate/issues/1258))

- The `sys` module is no longer automatically imported in `__main__` by
  reticulate.

- Fixed an issue on Windows where reticulate would fail to find Python
  installations from pyenv installed via scoop.

- Fixed an issue where
  [`configure_environment()`](https://rstudio.github.io/reticulate/dev/reference/configure_environment.md)
  would error on Windows.
  ([\#1247](https://github.com/rstudio/reticulate/issues/1247))

- Updated docs for compatibility with HTML5 / R 4.2.

- Updated r_to_py.sparseMatrix() method for compatibility with Matrix
  1.4-2.

## reticulate 1.25

CRAN release: 2022-05-11

- Fixed an issue where reticulate would fail if R was running embedded
  under rpy2. reticulate now ensures the Python GIL is acquired before
  calling into Python.
  ([\#1188](https://github.com/rstudio/reticulate/issues/1188),
  [\#1203](https://github.com/rstudio/reticulate/issues/1203))

- Fixed an issue where reticulate would fail to bind to an ArcGIS Pro
  conda environment
  ([\#1200](https://github.com/rstudio/reticulate/issues/1200),
  [@philiporlando](https://github.com/philiporlando)).

- Fixed an issue where reticulate would fail to bind to an Anaconda base
  environment on Windows.

- All commands that create, modify, or delete a Python environment now
  echo the system command about to be executed. Affected:
  virtualenv\_{create,install,remove}
  conda\_{create,clone,remove,install,update} py_install

- [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  and `create_virtualenv()` gain the ability to automatically select the
  latest patch of a requested Python version. e.g.:
  `install_python("3.9:latest")`,
  `create_virtualenv("my-env", version = "3.9:latest")`

- [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  `version` arg gains default value of `"3.9:latest"`.
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  can now be called with no arguments.

- Fixed an issue where reticulate would fail to bind to a conda python
  if the user didn’t have write permissions to the conda installation
  ([\#1156](https://github.com/rstudio/reticulate/issues/1156)).

- Fixed an issue where reticulate would fail to bind to a conda python
  if spaces were present in the file path to the associated conda binary
  ([\#1154](https://github.com/rstudio/reticulate/issues/1154)).

- `use_python(, required = TRUE)` now issues a warning if the request
  will be ignored
  ([\#1150](https://github.com/rstudio/reticulate/issues/1150)).

- New function
  [`py_repr()`](https://rstudio.github.io/reticulate/dev/reference/py_str.md)
  ([\#1157](https://github.com/rstudio/reticulate/issues/1157))

- [`print()`](https://rdrr.io/r/base/print.html) and related changes
  ([\#1148](https://github.com/rstudio/reticulate/issues/1148),
  [\#1157](https://github.com/rstudio/reticulate/issues/1157)):

  - The default [`print()`](https://rdrr.io/r/base/print.html) method
    for Python objects now invokes
    [`py_repr()`](https://rstudio.github.io/reticulate/dev/reference/py_str.md)
    instead of [`str()`](https://rdrr.io/r/utils/str.html).
  - All Python objects gain a default
    [`format()`](https://rdrr.io/r/base/format.html) method that invokes
    [`py_str()`](https://rstudio.github.io/reticulate/dev/reference/py_str.md).
  - [`py_str()`](https://rstudio.github.io/reticulate/dev/reference/py_str.md)
    default method no longer strips the object memory address.
  - [`print()`](https://rdrr.io/r/base/print.html) now returns the
    printed object invisibly, for composability with `%>%`.

- Exception handling changes
  ([\#1142](https://github.com/rstudio/reticulate/issues/1142),
  [@t-kalinowski](https://github.com/t-kalinowski)):

  - R error messages from Python exceptions are now truncated
    differently to satisfy `getOption("warning.length")`. A hint to call
    [`reticulate::py_last_error()`](https://rstudio.github.io/reticulate/dev/reference/py_last_error.md)
    is shown if the exception message was truncated.

  - Python buffers `sys.stderr` and `sys.stdout` are now flushed when
    Python exceptions are raised.

  \-[`py_last_error()`](https://rstudio.github.io/reticulate/dev/reference/py_last_error.md):

  - Return object is now an S3 object ‘py_error’, includes a default
    print method.

  - The python Exception object (‘python.builtin.Exception’) is
    available as an R attribute.

  - Gains the ability to restore a previous exception if provided in a
    call `py_last_error(previous_error)`

  - Python traceback objects gain a default
    [`format()`](https://rdrr.io/r/base/format.html) S3 method.

- Fixed
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  for scipy matrices when scipy \>= 1.8.0, since sparse matrices are now
  deprecated.

- Fixed
  [`r_to_py()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  for small scipy matrices.

- New maintainer: Tomasz Kalinowski

## reticulate 1.24

CRAN release: 2022-01-26

- Fixed an issue where `reticulate` would fail to bind to the system
  version of Python on macOS if command line tools were installed, but
  Xcode was not.

## reticulate 1.23

CRAN release: 2022-01-14

- [`use_condaenv()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md)
  gains the ability to accept an absolute path to a python binary for
  `envname`.

- All python objects gain a
  [`length()`](https://rdrr.io/r/base/length.html) method, that returns
  either `py_len(x)`, or if that fails, `as.integer(py_bool(x))`.

- [`conda_create()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  default for `python_version` changed from `NULL` to
  `miniconda_python_version()` (presently, 3.8).

- New function
  [`py_bool()`](https://rstudio.github.io/reticulate/dev/reference/py_bool.md),
  for evaluating Python “truthiness” of an object.

- `reticulate` gains the function
  [`py_list_packages()`](https://rstudio.github.io/reticulate/dev/reference/py_list_packages.md),
  and can be used to list the Python modules available and installed in
  a particular Python environment.
  ([\#933](https://github.com/rstudio/reticulate/issues/933))

- `reticulate` now supports conversion of Python
  [datatable](https://github.com/h2oai/datatable) objects.
  ([\#1081](https://github.com/rstudio/reticulate/issues/1081))

- [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md)
  gains support for invoking select magic and system commands like `!ls`
  and `%cd <dir>`. See `?repl_python()` for details and examples.

- The development branch for `reticulate` has moved to the “main”
  branch.

- `reticulate` gains
  [`reticulate::conda_update()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md),
  for updating the version of `conda` in a particular `conda`
  installation.

- `reticulate` gains
  [`reticulate::miniconda_uninstall()`](https://rstudio.github.io/reticulate/dev/reference/miniconda_uninstall.md),
  for uninstalling the reticulate-managed version of Miniconda.
  ([\#1077](https://github.com/rstudio/reticulate/issues/1077))

- [`reticulate::use_python()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md)
  and friends now assume `required = TRUE` by default. For backwards
  compatibility, when
  [`use_python()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md)
  is called as part of a package load hook, the default value will
  instead be `FALSE`.

- `reticulate` now provides support for Python environments managed by
  [poetry](https://python-poetry.org/). For projects containing a
  `pyproject.toml` file, `reticulate` will attempt to find and use the
  virtual environment managed by Poetry for that project.
  ([\#1031](https://github.com/rstudio/reticulate/issues/1031))

- The default version of Python used for the `r-reticulate` Miniconda
  environment installed via
  [`reticulate::install_miniconda()`](https://rstudio.github.io/reticulate/dev/reference/install_miniconda.md)
  has changed from 3.6 to 3.8.

- [`reticulate::install_miniconda()`](https://rstudio.github.io/reticulate/dev/reference/install_miniconda.md)
  now prefers installing the latest arm64 builds of miniforge. See
  <https://conda-forge.org/blog/posts/2020-10-29-macos-arm64/> for more
  details.

- [`reticulate::conda_create()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  gains the `environment` argument, used when creating a new conda
  environment based on an exported environment definition
  (e.g. `environment.yml` or `environment.json`).

- `reticulate` gains the function,
  [`conda_export()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md),
  for exporting a conda environment definition as YAML. Environments are
  exported as via the `conda env export` command.
  ([\#779](https://github.com/rstudio/reticulate/issues/779))

- `reticulate::find_conda()` will now locate miniforge Conda
  installations located within the default install locations.

- Fixed an issue that caused `reticulate::conda_install(pip = TRUE)` to
  fail on windows.
  ([\#1053](https://github.com/rstudio/reticulate/issues/1053),
  [@t-kalinowski](https://github.com/t-kalinowski))

## reticulate 1.22

CRAN release: 2021-09-17

- Fixed a regression that caused `reticulate::conda_install(pip = TRUE)`
  to fail. ([\#1052](https://github.com/rstudio/reticulate/issues/1052))

## reticulate 1.21

CRAN release: 2021-09-14

- `use_condaenv("base")` can now be used to activate the base Anaconda
  environment.

- `reticulate` will now execute any hooks registered via
  `setHook("reticulate.onPyInit", <...>)` after Python has been
  initialized. This can be useful for packages that need to take some
  action after `reticulate` has initialized Python.

- Further refined interrupt handling.

- Fixed an issue where attempting to bind `reticulate` to
  `/usr/bin/python3` on macOS could fail if Xcode was not installed.
  ([\#1017](https://github.com/rstudio/reticulate/issues/1017))

- The `reticulate` Python REPL no longer exits when a top-level
  interrupt is sent (e.g. via Ctrl + C).

- The miniconda auto-installer now supports aarch64 Linux machines.
  ([\#1012](https://github.com/rstudio/reticulate/issues/1012))

- Fixed an issue where matplotlib plots were incorrectly overwritten
  when multiple Python chunks in the same R Markdown document included
  plot output.
  ([\#1010](https://github.com/rstudio/reticulate/issues/1010))

- `reticulate` can now use the version of Python configured in projects
  using [pipenv](https://pypi.org/project/pipenv/). If the project
  contains a `Pipfile` at the root directory (as understood by
  [`here::here()`](https://here.r-lib.org/reference/here.html)), then
  `reticulate` will invoke `pipenv --venv` to determine the path to the
  Python virtual environment associated with the project. Note that the
  `RETICULATE_PYTHON` environment variable, as well as usages of
  `use_python(..., force = TRUE)`, will still take precedence.
  ([\#1006](https://github.com/rstudio/reticulate/issues/1006))

- Fixed an issue where `reticulate::py_run_string(..., local = TRUE)`
  failed to return the dictionary of defined Python objects in some
  cases.

## reticulate 1.20

CRAN release: 2021-05-03

- Fixed an issue causing tests to fail on CRAN’s M1mac machine.

## reticulate 1.19

CRAN release: 2021-04-21

- Fixed an issue where `reticulate`’s interrupt handlers could cause
  issues with newer versions of Python.

- `reticulate` now better handles Pandas categorical variables
  containing `NA` values.
  ([\#942](https://github.com/rstudio/reticulate/issues/942))

- `reticulate` now supports converting `pandas.NA` objects into R `NA`
  objects. ([\#950](https://github.com/rstudio/reticulate/issues/950))

- `reticulate` now sets the `PYTHONIOENCODING` environment variable to
  UTF-8 when running within RStudio. This should allow UTF-8 input and
  output to be handled more appropriately.

- `reticulate` gains the
  [`install_python()`](https://rstudio.github.io/reticulate/dev/reference/install_python.md)
  function, used to install different versions of Python via
  [pyenv](https://github.com/pyenv/pyenv)
  ([pyenv-windows](https://github.com/pyenv-win/pyenv-win) on Windows).

- Interrupt signals (e.g. those generated by `Ctrl + C`) are now better
  handled by `reticulate`. In particular, when
  [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md)
  is active, `Ctrl + C` can be used to interrupt a pending Python
  computation.

- [`virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  gains the `pip_version` and `setuptools_version` arguments, allowing
  users to control the versions of `pip` and `setuptools` used when
  initializing the virtual environment. The `extra` argument can also
  now be used to pass arbitrary command line arguments when necessary.

- [`virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  gains the `module` argument, used to control whether `virtualenv` or
  `venv` is used to create the requested virtual environment.

- `py_to_r.datetime.datetime` no longer errs when `tzname` is `NULL`,
  and instead assumes the time is formatted for `UTC`.
  ([\#876](https://github.com/rstudio/reticulate/issues/876))

- `reticulate` now supports the rendering of
  [plotly](https://plotly.com/) plots and
  [Altair](https://altair-viz.github.io/) charts in rendered R Markdown
  documents. ([\#711](https://github.com/rstudio/reticulate/issues/711))

- `reticulate` now avoids invoking property methods when inferring the
  type for Python class members, for auto-completion systems.
  ([\#907](https://github.com/rstudio/reticulate/issues/907))

- `reticulate` now attempts to set the `QT_QPA_PLATFORM_PLUGIN_PATH`
  environment variable when initializing a Conda installation of Python,
  when that associated plugins directory exists.
  ([\#586](https://github.com/rstudio/reticulate/issues/586))

- The `reticulate` Python engine now supports the `results = "hold"`
  knitr chunk option. When set, any generated outputs are “held” and
  then displayed after the associated chunk’s source code.
  ([\#530](https://github.com/rstudio/reticulate/issues/530))

- [`conda_create()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  gains the `python_version` argument, making it easier to request that
  Conda environments are created with a pre-specified version of Python.
  ([\#766](https://github.com/rstudio/reticulate/issues/766))

- Fixed an issue where
  [`reticulate::conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  would attempt to re-install the default Python package, potentially
  upgrading or downgrading the version of Python used in an environment.

- Fixed an issue where `reticulate` invoked its `reticulate.initialized`
  hook too early.

- Fixed an issue where Python modules loaded on a separate thread could
  cause a crash.
  ([\#885](https://github.com/rstudio/reticulate/issues/885))

- [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  now allows version specifications for the `python_version` argument;
  e.g. `conda_install(python_version = ">=3.6")`.
  ([\#880](https://github.com/rstudio/reticulate/issues/880))

- Fixed an issue where
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  failed to pass along `forge` and `channel` in calls to
  [`conda_create()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md).
  ([\#878](https://github.com/rstudio/reticulate/issues/878))

- Fixed an issue where Python’s auto-loader hooks could fail when
  binding to a Python 2.7 installation.

## reticulate 1.18

CRAN release: 2020-10-25

- Fixed an issue where `python_config()` could throw an error when
  attempting to query information about a Python 2.6 installation.

## reticulate 1.17

CRAN release: 2020-10-21

- `reticulate` now checks for and disallows installation of Python
  packages during `R CMD check`.

- `reticulate` no longer injects the `r` helper object into the main
  module if another variable called `r` has already been defined.

- The function
  [`py_help_handler()`](https://rstudio.github.io/reticulate/dev/reference/py_help_handler.md)
  has now been exported, to be used by front-ends and other tools which
  need to provide help for Python objects in different contexts.
  ([\#864](https://github.com/rstudio/reticulate/issues/864))

- Fixed an issue where timezone information could be lost when
  converting Python datetime objects to R.
  ([\#829](https://github.com/rstudio/reticulate/issues/829))

- Fixed an issue where numeric (rather than integer) dimensions could
  cause issues when converting SciPy sparse matrices to their R
  counterparts.
  ([\#844](https://github.com/rstudio/reticulate/issues/844))

- Fixed an issue where R `data.frame`s with non-ASCII column names could
  not be converted to Pandas DataFrames.
  ([\#834](https://github.com/rstudio/reticulate/issues/834))

- Fixed an issue where the `pip_ignore_installed` argument in
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  was silently being ignored.

- Fixed an issue where
  [`reticulate::conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  could re-install Python into an environment when not explicitly
  requested by the user.

- `reticulate` now sets `LD_LIBRARY_PATH` when discovering Python.
  ([\#836](https://github.com/rstudio/reticulate/issues/836))

- `reticulate` is now better at capturing Python logger streams (those
  that write to stdout or stderr) when
  [`py_capture_output()`](https://rstudio.github.io/reticulate/dev/reference/py_capture_output.md)
  is set. ([\#825](https://github.com/rstudio/reticulate/issues/825))

- `reticulate` no longer calls
  [`utils::loadhistory()`](https://rdrr.io/r/utils/savehistory.html)
  after each REPL iteration.

- `reticulate` now better detects when Python modules are loaded.

- [`reticulate::import_from_path()`](https://rstudio.github.io/reticulate/dev/reference/import.md)
  now accepts the `delay_load` parameter, allowing modules which should
  be loaded from a pre-specified path to be lazy-loaded.

- Fixed an issue where `reticulate` load hooks (normally defined via
  `setHook("reticulate::<module>::load", ...)`) would segfault if those
  hooks attempted to load the hooked module.

- `reticulate` now attempts to resolve the conda binary used to create
  the associated Conda environment in calls to
  [`py_install()`](https://rstudio.github.io/reticulate/dev/reference/py_install.md).
  This should fix use cases where Conda environments are placed outside
  of the Conda installation itself.

- `reticulate` now sets `PYTHONPATH` before loading Python, to ensure
  modules are looked up in the same locations where a regular Python
  interpreter would find them on load. This should fix issues where
  `reticulate` was unable to bind to a Python virtual environment in
  some cases.

- [`reticulate::virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  gains the `packages` argument, allowing one to choose a set of
  packages to be installed (via `pip install`) after the virtual
  environment has been created.

- [`reticulate::virtualenv_create()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  gains the `system_site_packages` argument, allowing one to control
  whether the `--system-site-packages` flag is passed along when
  creating a new virtual environment. The default value can be
  customized via the `"reticulate.virtualenv.system_site_packages"`
  option and now defaults to `FALSE` when unset.

- Fixed an issue where
  [`reticulate::configure_environment()`](https://rstudio.github.io/reticulate/dev/reference/configure_environment.md)
  would fail when attempting to configure an Anaconda environment.
  ([\#794](https://github.com/rstudio/reticulate/issues/794))

- `reticulate` now avoids presenting a Miniconda prompt for interactive
  sessions during R session initialization.

- Fixed unsafe usages of `Rprintf()` and `REprintf()`.

- [`reticulate::py_install()`](https://rstudio.github.io/reticulate/dev/reference/py_install.md)
  better respects the `method` argument, when
  [`py_install()`](https://rstudio.github.io/reticulate/dev/reference/py_install.md)
  is called without an explicit environment name.
  ([\#777](https://github.com/rstudio/reticulate/issues/777))

- `reticulate:::pip_freeze()` now better handles `pip` direct
  references.
  ([\#775](https://github.com/rstudio/reticulate/issues/775))

- Fixed an issue where output generated from
  [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md)
  would be buffered until the whole submitted command had completed.
  ([\#739](https://github.com/rstudio/reticulate/issues/739),
  [@randy3k](https://github.com/randy3k))

- `reticulate` now explicitly qualifies symbols used from TinyThread
  with `tthread::`, to avoid issues with symbol conflicts during
  compilation.
  ([\#773](https://github.com/rstudio/reticulate/issues/773))

- `reticulate` will now prefer an existing Miniconda installation over a
  `conda` binary on the PATH, when looking for Conda.
  ([\#790](https://github.com/rstudio/reticulate/issues/790))

## reticulate 1.16

CRAN release: 2020-05-27

- TinyThread now calls `Rf_error()` rather than `std::terminate()` when
  an internal error occurs.

- Conversion of Pandas DataFrames to R no longer emits deprecation
  warnings with pandas \>= 0.25.0.
  ([\#762](https://github.com/rstudio/reticulate/issues/762))

- `reticulate` now properly handles the version strings returned by beta
  versions of `pip`.
  ([\#757](https://github.com/rstudio/reticulate/issues/757))

- [`conda_create()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  gains the `forge` and `channel` arguments, analogous to those already
  in
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md).
  ([\#752](https://github.com/rstudio/reticulate/issues/752),
  [@jtilly](https://github.com/jtilly))

## reticulate 1.15

CRAN release: 2020-04-02

- `reticulate` now ensures SciPy `csr_matrix` objects are sorted before
  attempting to convert them to their R equivalent.
  ([\#738](https://github.com/rstudio/reticulate/issues/738),
  [@paulofelipe](https://github.com/paulofelipe))

- Fixed an issue where calling `input()` from Python with no prompt
  would fail.
  ([\#728](https://github.com/rstudio/reticulate/issues/728))

- Lines ending with a semi-colon are no longer auto-printed in the
  `reticulate` REPL.
  ([\#717](https://github.com/rstudio/reticulate/issues/717),
  [@jsfalk](https://github.com/jsfalk))

- `reticulate` now searches for Conda binaries in /opt/anaconda and
  /opt/miniconda.
  ([\#713](https://github.com/rstudio/reticulate/issues/713))

- The `conda` executable used by `reticulate` can now be configured
  using an R option. Use `options(reticulate.conda_binary = <...>)` to
  force `reticulate` to use a particular `conda` executable.

- [`reticulate::use_condaenv()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md)
  better handles cases where no matching environment could be found.
  ([\#687](https://github.com/rstudio/reticulate/issues/687))

- `reticulate` gains the
  [`py_ellipsis()`](https://rstudio.github.io/reticulate/dev/reference/py_ellipsis.md)
  function, used to access the Python `Ellipsis` builtin.
  ([\#700](https://github.com/rstudio/reticulate/issues/700),
  [@skeydan](https://github.com/skeydan))

- [`reticulate::configure_environment()`](https://rstudio.github.io/reticulate/dev/reference/configure_environment.md)
  now only allows environment configuration within interactive R
  sessions, and ensures that the version of Python that has been
  initialized by Python is indeed associated with a virtual environment
  or Conda environment. Use
  `reticulate::configure_environment(force = TRUE)` to force environment
  configuration within non-interactive R sessions.

- `reticulate` now automatically flushes output written to Python’s
  stdout / stderr, as a top-level task added by
  [`addTaskCallback()`](https://rdrr.io/r/base/taskCallback.html). This
  behavior is controlled with the `options(reticulate.autoflush)`
  option. ([\#685](https://github.com/rstudio/reticulate/issues/685))

- [`reticulate::install_miniconda()`](https://rstudio.github.io/reticulate/dev/reference/install_miniconda.md)
  no longer attempts to modify the system PATH or registry when
  installing Miniconda.
  ([\#681](https://github.com/rstudio/reticulate/issues/681))

- [`reticulate::conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  gains the `channel` argument, allowing custom Conda channels to be
  used when installing Python packages.
  ([\#443](https://github.com/rstudio/reticulate/issues/443))

- [`reticulate::configure_environment()`](https://rstudio.github.io/reticulate/dev/reference/configure_environment.md)
  can now be used to configure a non-Miniconda Python environment.
  ([\#682](https://github.com/rstudio/reticulate/issues/682);
  [@skeydan](https://github.com/skeydan))

- Fixed an issue where matplotlib plots would be included using absolute
  paths, which fails in non-standalone documents rendered to HTML.
  ([\#669](https://github.com/rstudio/reticulate/issues/669))

- Fixed an issue where `reticulate` would attempt to flush a
  non-existent stdout / stderr stream.
  ([\#584](https://github.com/rstudio/reticulate/issues/584))

## reticulate 1.14

CRAN release: 2019-12-17

- Fixed an issue where
  [`rmarkdown::render()`](https://pkgs.rstudio.com/rmarkdown/reference/render.html)
  could fail when including matplotlib plots when `knit_root_dir` is
  set. ([\#645](https://github.com/rstudio/reticulate/issues/645))

- `reticulate` now scans for Conda installations within the ~/opt
  folder, as per the updated installers distributed for macOS.
  ([\#661](https://github.com/rstudio/reticulate/issues/661))

- Python classes can now be defined directly from R using the
  [`PyClass()`](https://rstudio.github.io/reticulate/dev/reference/PyClass.md)
  function. ([\#635](https://github.com/rstudio/reticulate/issues/635);
  [@dfalbel](https://github.com/dfalbel))

- reticulate is now compatible with Python 3.9.
  ([\#630](https://github.com/rstudio/reticulate/issues/630),
  [@skeydan](https://github.com/skeydan))

- Pandas DataFrames with a large number of columns should now be
  converted to R data.frames more quickly.
  ([\#620](https://github.com/rstudio/reticulate/issues/620),
  [@skeydan](https://github.com/skeydan))

- Python loggers are now better behaved in the Python chunks of R
  Markdown documents.
  ([\#386](https://github.com/rstudio/reticulate/issues/386))

- reticulate will now attempt to bind to `python3` rather than `python`,
  when no other version of Python has been explicitly requested by
  e.g. [`use_python()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md).

- reticulate now provides R hooks for Python’s `input()` and
  `raw_input()` functions. It should now be possible to read user input
  from Python scripts loaded by reticulate.
  ([\#610](https://github.com/rstudio/reticulate/issues/610))

- `reticulate` now more consistently normalizes the paths reported by
  [`py_config()`](https://rstudio.github.io/reticulate/dev/reference/py_config.md).
  ([\#609](https://github.com/rstudio/reticulate/issues/609))

- `reticulate` now provides a mechanism for allowing client packages to
  declare their Python package dependencies. Packages should declare the
  Python packages they require as part of the `Config/reticulate` field
  in their `DESCRIPTION` file. Currently, this only activated when using
  Miniconda; as the assumption is that users will otherwise prefer to
  manually manage their Python environments. Please see
  [`vignette("python_dependencies")`](https://rstudio.github.io/reticulate/dev/articles/python_dependencies.md)
  for more details.

- `reticulate` will now prompt the user to create and use a
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  environment when no other suitable Python environment has already been
  requested. This should help ease some of the trouble in setting up a
  Python environment on different platforms. The installer code was
  contributed by [@hafen](https://github.com/hafen), from the
  [rminiconda](https://github.com/hafen/rminiconda) package.

- Fixed an issue where `virtualenv_create(..., python = "<python>")`
  could fail to use the requested version of Python when `venv` is not
  installed. ([\#399](https://github.com/rstudio/reticulate/issues/399))

- Fixed an issue where iterable Python objects could not be iterated
  with
  [`iter_next()`](https://rstudio.github.io/reticulate/dev/reference/iterate.md)
  due to a missing class.
  ([\#603](https://github.com/rstudio/reticulate/issues/603))

- Fixed an issue where Conda environments could be mis-detected as
  virtual environments.

- R functions wrapping Python functions now inherit the formal arguments
  as specified by Python, making autocompletion more reliable.
  ([\#573](https://github.com/rstudio/reticulate/issues/573),
  [@flying-sheep](https://github.com/flying-sheep))

- Fixed an issue where attempts to query Conda for environments could
  fail on Windows.
  ([\#576](https://github.com/rstudio/reticulate/issues/576);
  [\#575](https://github.com/rstudio/reticulate/issues/575);
  [@dfalbel](https://github.com/dfalbel))

- Properly check for NULL keyword arguments in `call_r_function()`.
  ([\#562](https://github.com/rstudio/reticulate/issues/562),
  [@dfalbel](https://github.com/dfalbel))

## reticulate 1.13

CRAN release: 2019-07-24

- Fixed an issue where subsetting with `[.python.builtin.object` could
  fail when `convert = TRUE` is set on the associated Python object.
  ([\#554](https://github.com/rstudio/reticulate/issues/554))

- Fixed an issue where the wrong definition of
  `[[.python.builtin.object` was being exported.
  ([\#554](https://github.com/rstudio/reticulate/issues/554))

- [`py_install()`](https://rstudio.github.io/reticulate/dev/reference/py_install.md)
  now accepts `python_version`, and can be used if a particular version
  of Python is required for a Conda environment. (This argument is
  ignored for virtual environments.)
  ([\#549](https://github.com/rstudio/reticulate/issues/549))

- Fixed an issue where reticulate could segfault in some cases
  (e.g. when using the
  [`iterate()`](https://rstudio.github.io/reticulate/dev/reference/iterate.md)
  function). ([\#551](https://github.com/rstudio/reticulate/issues/551))

- It is now possible to compile `reticulate` with support for debug
  versions of Python by setting the `RETICULATE_PYTHON_DEBUG`
  preprocessor define during compilation.
  ([\#548](https://github.com/rstudio/reticulate/issues/548))

- reticulate now warns if it did not honor the user’s request to load a
  particular version of Python, as through
  e.g. [`reticulate::use_python()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md).
  ([\#545](https://github.com/rstudio/reticulate/issues/545))

- [`py_save_object()`](https://rstudio.github.io/reticulate/dev/reference/py_save_object.md)
  and
  [`py_load_object()`](https://rstudio.github.io/reticulate/dev/reference/py_save_object.md)
  now accept `...` arguments.
  ([\#542](https://github.com/rstudio/reticulate/issues/542))

- [`py_install()`](https://rstudio.github.io/reticulate/dev/reference/py_install.md)
  has been revamped, and now better detects available Python tooling
  (virtualenv vs. venv vs. Conda).
  ([\#544](https://github.com/rstudio/reticulate/issues/544))

- reticulate now flushes stdout / stderr after calls to
  [`py_run_file()`](https://rstudio.github.io/reticulate/dev/reference/py_run.md)
  and
  [`py_run_string()`](https://rstudio.github.io/reticulate/dev/reference/py_run.md).

- Python tuples are now converted recursively, in the same way that
  Python lists are. This means that the sub-elements of the tuple will
  be converted to R objects when possible.
  ([\#525](https://github.com/rstudio/reticulate/issues/525),
  [@skeydan](https://github.com/skeydan))

- Python OrderedDict objects with non-string keys are now properly
  converted to R.
  ([\#516](https://github.com/rstudio/reticulate/issues/516))

- Fixed an issue where reticulate could crash after a failed attempt to
  load NumPy.
  ([\#497](https://github.com/rstudio/reticulate/issues/497),
  [@ecoughlan](https://github.com/ecoughlan))

## reticulate 1.12

CRAN release: 2019-04-12

- Fixed an issue where Python objects within Python lists would not be
  converted to R objects as expected.

- Fixed an issue where single-row data.frames with row names could not
  be converted.
  ([\#468](https://github.com/rstudio/reticulate/issues/468))

- Fixed an issue where `reticulate` could fail to query Anaconda
  environment names with Anaconda 3.7.

- Fixed an issue where vectors of R Dates were not converted correctly.
  ([\#454](https://github.com/rstudio/reticulate/issues/454))

- Fixed an issue where R Dates could not be passed to Python functions.
  ([\#458](https://github.com/rstudio/reticulate/issues/458))

## reticulate 1.11.1

CRAN release: 2019-03-06

- Fixed a failing virtual environment test on CRAN.

## reticulate 1.11

CRAN release: 2019-02-27

- Fixed an issue where attempts to activate virtual environments created
  with virtualenv 16.4.1 would fail.
  ([\#437](https://github.com/rstudio/reticulate/issues/437))

- Fixed an issue where conversion of Pandas Categorical variables to R
  objects would fail.
  ([\#389](https://github.com/rstudio/reticulate/issues/389))

- Textual output generated when adding items to a matplotlib plot object
  are now suppressed.

- If the last statement in a Python chunk returns a matplotlib plot
  object, the plot will now be auto-shown as in other environments.

- The reticulate function help handler now returns function arguments
  for Python builtin functions.

- Top-level Python statements can now include leading indent when
  submitted with
  [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md).

- The current `matplotlib` figure is now cleared as each Python chunk in
  an R Markdown document is run.

- The `r` helper object (used for evaluating R code from Python) now
  better handles conversion of R functions.
  ([\#383](https://github.com/rstudio/reticulate/issues/383))

- The
  [`use_virtualenv()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md)
  function now understands how to bind to virtual environments created
  by the Python `venv` module.

- Reticulate better handles conversions of R lists to Python, and
  similarly, Python lists to R. We now call
  [`r_to_py()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  on each sub-element of an R list, and similarly,
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  on each sub-element of a Python list.

- Reticulate now always converts R `Date` objects into Python `datetime`
  objects. Note that these conversions can be inefficient – if you would
  prefer conversion to NumPy `datetime64` objects / arrays, you should
  convert your date to `POSIXct` first.

- Python chunks containing errors will cause execution to halt if
  ‘error=FALSE’ during render, conforming with the default knitr
  behavior for R chunks.

- The output of bare statements (e.g. `1 + 1`) is now emitted as output
  when using the reticulate Python engine.

- Remapping of Python output streams to be R can now be explicitly
  enabled by setting the environment variable
  `RETICULATE_REMAP_OUTPUT_STREAMS` to 1.
  ([\#335](https://github.com/rstudio/reticulate/issues/335))

- Allow syntax errors in Python chunks with ‘eval = FALSE’
  ([\#343](https://github.com/rstudio/reticulate/issues/343))

- Avoid dropping blank lines in Python chunks
  ([\#328](https://github.com/rstudio/reticulate/issues/328))

- Use “agg” matplotlib backend when running under RStudio Desktop
  (avoids crashes when attempting to generate Python plots)

- Add [`as.character()`](https://rdrr.io/r/base/character.html) S3
  method for Python bytes (defaults to converting using UTF-8 encoding)

- Add
  [`py_main_thread_func()`](https://rstudio.github.io/reticulate/dev/reference/py_main_thread_func.md)
  for providing R callbacks to Python libraries that may invoke the
  function on a Python background thread.

- Add `py_to_r` S3 methods for Scipy sparse matrices: CSR to dgRMatrix,
  COO to dgTMatrix, and for all other sparse matrices, conversion via
  CSC/dgCMatrix.

## reticulate 1.10

CRAN release: 2018-08-05

- Output is now properly displayed when using the `reticulate` REPL with
  Windows + Python 2.7.

- Address memory protection issues identified by rchk

- Make variables defined using `%as%` operator in
  [`with()`](https://rdrr.io/r/base/with.html) available after execution
  of the with block (same behavior as Python).

- Check for presence of “**module**” property before reading in
  `as_r_class()`

- Only update pip in
  [`virtualenv_install()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  when version is \< 8.1

- Support converting Python `OrderedDict` to R

- Support for iterating all types of Python iterable

- Add
  [`conda_python()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  and
  [`virtualenv_python()`](https://rstudio.github.io/reticulate/dev/reference/virtualenv-tools.md)
  functions for finding the python binary associated with an
  environment.

## reticulate 1.9

CRAN release: 2018-07-06

- Detect python 3 in environments where there is no python 2
  (e.g. Ubuntu 18.04)

- Always call r_to_py S3 method when converting objects from Python to R

- Handle NULL module name when determining R class for Python objects

- Convert RAW vectors to Python bytearray; Convert Python bytearray to
  RAW

- Use importlib for detecting modules (rather than imp) for Python \>=
  3.4

- Close text connection used for reading Python configuration probe

## reticulate 1.8

CRAN release: 2018-06-08

- [`source_python()`](https://rstudio.github.io/reticulate/dev/reference/source_python.md)
  now flushes stdout and stderr after running the associated Python
  script, to ensure that
  [`print()`](https://rdrr.io/r/base/print.html)-ed output is output to
  the console.
  ([\#284](https://github.com/rstudio/reticulate/issues/284))

- Fixed an issue where logical R matrices would not be converted
  correctly to their NumPy counterpart.
  ([\#280](https://github.com/rstudio/reticulate/issues/280))

- Fixed an issue where Python chunks containing multiple statements on
  the same line would be evaluated and printed multiple times.

- Added
  [`py_get_item()`](https://rstudio.github.io/reticulate/dev/reference/py_get_item.md),
  [`py_set_item()`](https://rstudio.github.io/reticulate/dev/reference/py_get_item.md),
  and
  [`py_del_item()`](https://rstudio.github.io/reticulate/dev/reference/py_get_item.md)
  as lower-level APIs for directly accessing the items of e.g. a Python
  dictionary or a Pandas DataFrame.

- Fix issue with Pandas column names that clash with built in methods
  (e.g. ‘pop’)

- Improve default [`str()`](https://rdrr.io/r/utils/str.html) output for
  Python objects (print `__dict__` if available)

## reticulate 1.7

CRAN release: 2018-04-28

- Improved filtering of non-numeric characters in Python / NumPy
  versions.

- Added
  [`py_func()`](https://rstudio.github.io/reticulate/dev/reference/py_func.md)
  to wrap an R function in a Python function with the same signature as
  that of the original R function.

- Added support for conversion between `Matrix::dgCMatrix` objects in R
  and `Scipy` CSC matrices in Python.

- [`source_python()`](https://rstudio.github.io/reticulate/dev/reference/source_python.md)
  can now source a Python script from a URL into R environments.

- Always run
  [`source_python()`](https://rstudio.github.io/reticulate/dev/reference/source_python.md)
  in the main Python module.

- [`py_install()`](https://rstudio.github.io/reticulate/dev/reference/py_install.md)
  function for installing Python packages into virtualenvs and conda
  envs

- Automatically create conda environment for
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)

- Removed `delay_load` parameter from
  [`import_from_path()`](https://rstudio.github.io/reticulate/dev/reference/import.md)

## reticulate 1.6

CRAN release: 2018-03-21

- [`repl_python()`](https://rstudio.github.io/reticulate/dev/reference/repl_python.md)
  function implementing a lightweight Python REPL in R.

- Support for converting Pandas objects (`Index`, `Series`, `DataFrame`)

- Support for converting Python `datetime` objects.

- [`py_dict()`](https://rstudio.github.io/reticulate/dev/reference/dict.md)
  function to enable creation of dictionaries based on lists of keys and
  values.

- Provide default base directory (e.g. ‘~/.virtualenvs’) for
  environments specified by name in
  [`use_virtualenv()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md).

- Fail when environment not found with
  `use_condaenv(..., required = TRUE)`

- Ensure that `use_*` python version is satisfied when using
  [`eng_python()`](https://rstudio.github.io/reticulate/dev/reference/eng_python.md)

- Forward `required` argument from
  [`use_virtualenv()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md)
  and
  [`use_condaenv()`](https://rstudio.github.io/reticulate/dev/reference/use_python.md)

- Fix leak which occurred when assigning R objects into Python
  containers

- Add support for Conda Forge (enabled by default) to
  [`conda_install()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)

- Added functions for managing Python virtual environments (virtualenv)

## reticulate 1.5

CRAN release: 2018-02-14

- Remove implicit documentation extraction for Python classes

- Add `Library\bin` to PATH on Windows to ensure Anaconda can find MKL

- New
  [`source_python()`](https://rstudio.github.io/reticulate/dev/reference/source_python.md)
  function for sourcing Python scripts into R environments.

## reticulate 1.4

CRAN release: 2018-01-09

- Support for `RETICULATE_DUMP_STACK_TRACE` environment variable which
  can be set to the number of milliseconds in which to output into
  stderr the call stacks from all running threads.

- Provide hook to change target module when delay loading

- Scan for conda environments in system-level installations

- Support for miniconda environments

- Implement `eval`, `echo`, and `include` knitr chunk options for Python
  engine

## reticulate 1.3.1

CRAN release: 2017-11-24

- Bugfix: ensure single-line Python chunks that produce no output still
  have source code emitted.

## reticulate 1.3

CRAN release: 2017-11-16

- Use existing instance of Python when reticulate is loaded within an
  embedded Python environment (e.g. rpy2, rice, etc.)

- Force use of Python specified in PYTHON_SESSION_INITIALIZED (defined
  by rpy2)

- Define R_SESSION_INITIALIZED (used by rpy2)

- Force use of Python when `required = TRUE` in `use_python` functions

- Force use of Python specified by RETICULATE_PYTHON

- `dict`: Don’t scan parent frame for Python objects if a single unnamed
  list is passed.

- Wait as long as required for scheduling generator calls on the main
  thread

- Refine stripping of object addresses from output of
  [`py_str()`](https://rstudio.github.io/reticulate/dev/reference/py_str.md)
  method

- Added
  [`py_id()`](https://rstudio.github.io/reticulate/dev/reference/py_id.md)
  function to get globally unique ids for Python objects

- Added
  [`py_len()`](https://rstudio.github.io/reticulate/dev/reference/py_len.md)
  function and S3 [`length()`](https://rdrr.io/r/base/length.html)
  method for Python lists (already had
  [`length()`](https://rdrr.io/r/base/length.html) methods for dicts,
  tuples, and NumPy arrays).

- Exported `py` object (reference to Python main module)

- Added
  [`eng_python()`](https://rstudio.github.io/reticulate/dev/reference/eng_python.md)
  (knitr engine for Python chunks)

- Improved compatibility with strings containing high unicode characters
  when running under Python 2

- Remove `dim` methods for NumPy arrays (semantics of NumPy reshaping
  are different from R reshaping)

- Added `array_reshape` function for reshaping R arrays using NumPy
  (row-major) semantics.

- Provide mechanism for custom R wrapper objects for Python objects

- Added interface to pickle
  ([`py_save_object()`](https://rstudio.github.io/reticulate/dev/reference/py_save_object.md)
  and
  [`py_load_object()`](https://rstudio.github.io/reticulate/dev/reference/py_save_object.md))

- Catch and print errors which occur in generator functions

- Write using Rprintf when providing custom Python output streams
  (enables correct handling of terminal control characters)

- Implement `isatty` when providing custom Python output streams

## reticulate 1.2

CRAN release: 2017-09-27

- Add `np_array` function for creating NumPy arrays and converting the
  data type, dimensions, and in-memory ordering of existing NumPy
  arrays.

- Add `dim` and `length` functions for NumPy arrays

- Add `py_set_seed` function for setting Python and NumPy random seeds.

- Search in additional locations for Anaconda on Linux/Mac

- Improved support for UTF-8 conversions (always use UTF-8 when
  converting from Python to R)

- Ignore private (“\_” prefixed) attributes of dictionaries for
  .DollarNames

- Provide “\`function\`” rather than “function” in completions.

- Fail gracefully if call to conda in `conda_list` results in an error

- Add `pip_ignore_installed` option to `conda_install` function.

## reticulate 1.1

CRAN release: 2017-08-26

- Allow
  [`dict()`](https://rstudio.github.io/reticulate/dev/reference/dict.md)
  function to accept keys with mixed alpha/numeric characters

- Use
  [`conda_list()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  to discover conda environments on Windows (slower but much more
  reliable than scanning the filesystem)

- Add interface for registering F1 help handlers for Python modules

- Provide virtual/conda env hint mechanism for delay loaded imports

## reticulate 1.0

CRAN release: 2017-07-25

- Search WORKON_HOME (used by virtualenv_wrapper) for Python
  environments

- Support `priority` field for delay loaded modules.

- Use json output from conda_list (handle spaces in path of conda env)

- Look for callable before iterable when converting Python objects to R

- Correct propagation of errors in R functions called from Python

- Support for generators (creating Python iterators from R functions)

- Changed default `completed` value for
  [`iter_next()`](https://rstudio.github.io/reticulate/dev/reference/iterate.md)
  to `NULL` (was `NA`)

- Support for converting 16-bit floats (NPY_HALF) to R

- Don’t throw error when probing Python \<= 2.6

- Copy Python dictionary before converting to R named list (fixes issue
  with dictionaries that are mutated during iteration, e.g. sys.modules)

- Ensure that existing warning filters aren’t reset by
  py_suppress_warnings

## reticulate 0.9

CRAN release: 2017-06-23

- Detect older versions of Anaconda during registry scanning.

- Don’t probe python versions on windows when no executable is found

- Poll for interrupts every 500ms rather than 100ms

- Provide sys.stdout and sys.stderr when they are None (e.g. in R GUI)

- Add Scripts directory to PATH on Windows

- Add iter_next function for element-by-element access to iterators

- Eliminate special print method for iterators/generators

- Added
  [`py_help()`](https://rstudio.github.io/reticulate/dev/reference/py_help.md)
  function for printing documentation on Python objects

- Added
  [`conda_version()`](https://rstudio.github.io/reticulate/dev/reference/conda-tools.md)
  function.

- Search
  [`dict()`](https://rstudio.github.io/reticulate/dev/reference/dict.md)
  parent frames for symbols; only use symbols which inherit from
  python.builtin.object as keys.

## reticulate 0.8

CRAN release: 2017-05-22

- Add
  [`import_from_path()`](https://rstudio.github.io/reticulate/dev/reference/import.md)
  function for importing Python modules from the filesystem.

- Add
  [`py_discover_config()`](https://rstudio.github.io/reticulate/dev/reference/py_discover_config.md)
  function to determine which versions of Python will be discovered and
  which one will be used by reticulate.

- Add
  [`py_function_docs()`](https://rstudio.github.io/reticulate/dev/reference/py_function_wrapper.md)
  amd
  [`py_function_wrapper()`](https://rstudio.github.io/reticulate/dev/reference/py_function_wrapper.md)
  utility functions for scaffolding R wrappers for Python functions.

- Add
  [`py_last_error()`](https://rstudio.github.io/reticulate/dev/reference/py_last_error.md)
  function for retrieving last Python error.

- Convert 0-dimension NumPy arrays (scalars) to single element R vectors

- Convert “callable” Python objects to R functions

- Automatically add Python bin directory to system PATH for consistent
  version usage in reticulate and calls to system

- Added [`length()`](https://rdrr.io/r/base/length.html) method for
  tuple objects

- Enable specification of `__name__` for R functions converted to Python
  functions.

- Give priority to the first registered delay load module (previously
  the last registered module was given priority)

- Add additional safety checks to detect use of NULL xptr objects
  (i.e. objects from a previous session). This should mean that S3
  methods no longer need to check whether they are handling an xptr.

- Added
  [`py_eval()`](https://rstudio.github.io/reticulate/dev/reference/py_eval.md)
  function for evaluating simple Python statements.

- Add `local` option to
  [`py_run_string()`](https://rstudio.github.io/reticulate/dev/reference/py_run.md)
  and
  [`py_run_file()`](https://rstudio.github.io/reticulate/dev/reference/py_run.md).
  Modify behavior to return local execution dictionary (rather than a
  reference to the main module).

- Use `PyImport_Import` rather than `PyImport_ImportModule` for
  [`import()`](https://rstudio.github.io/reticulate/dev/reference/import.md)

- Added ability to customize mapping of Python classes to R classes via
  the `as` argument to
  [`import()`](https://rstudio.github.io/reticulate/dev/reference/import.md)
  and the
  [`register_class_filter()`](https://rstudio.github.io/reticulate/dev/reference/register_class_filter.md)
  function

- Added separate `on_load` and `on_error` functions for `delay_load`

- Scan customary root directories for virtualenv installations

- Allow calling `__getitem__` via `[[` operator (zero-based to match
  Python style indexing)

- Added `conda_*` family of functions for using conda utilities from
  within R.

- Implement comparison operators (e.g. `==`, `>=`, etc.) for Python
  objects

- Implement [`names()`](https://rdrr.io/r/base/names.html) generic for
  Python objects

- Improve performance for marshalling of large Python dictionaries and
  iterators that return large numbers of items.

- Implement `str` methods for Python List, Dict, and Tuple (to prevent
  printing of very large collections via default `str` method)

- Use [`grepl()`](https://rdrr.io/r/base/grep.html) rather than
  [`endsWith()`](https://rdrr.io/r/base/startsWith.html) for
  compatibility with R \<= 3.2

- Use `inspect.getmro` rather than `__bases__` for enumerating the base
  classes of Python objects.

- Fix `PROTECT`/`UNPROTECT` issue detected by CRAN

- Correct conversion of strings with Unicode characters on Windows

- Fix incompatibility with system-wide Python installations on Windows

- Fix issue with Python dictionary keys that shared names with primitive
  R functions (don’t check environment inheritance chain when looking
  for dictionary key objects by name).

- Propagate `convert` parameter for modules with `delay_load`

## reticulate 0.7

CRAN release: 2017-03-14

- Initial CRAN release
