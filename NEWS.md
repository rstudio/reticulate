
## reticulate 1.17 (UNRELEASED)

- Removed `py_function_docs()`, `py_function_wrapper()`, and
  `py_function_custom_scaffold()` from `reticulate` as they
  have been migrated to the [scaffolder package](https://github.com/terrytangyuan/scaffolder). (#864)

- Fixed an issue where timezone information could be lost when converting
  Python datetime objects to R. (#829)

- Fixed an issue where numeric (rather than integer) dimensions could cause
  issues when converting SciPy sparse matrices to their R counterparts. (#844)

- Fixed an issue where R `data.frame`s with non-ASCII column names could not
  be converted to Pandas DataFrames. (#834)

- Fixed an issue where the `pip_ignore_installed` argument in `conda_install()`
  was silently being ignored.

- Fixed an issue where `reticulate::conda_install()` could re-install Python
  into an environment when not explicitly requested by the user.
  
- `reticulate` now sets `LD_LIBRARY_PATH` when discovering Python. (#836)

- `reticulate` is now better at capturing Python logger streams (those that
  write to stdout or stderr) when `py_capture_output()` is set. (#825)

- `reticulate` no longer calls `utils::loadhistory()` after each REPL iteration.

- `reticulate` now better detects when Python modules are loaded.

- `reticulate::import_from_path()` now accepts the `delay_load` parameter,
  allowing modules which should be loaded from a pre-specified path
  to be lazy-loaded.

- Fixed an issue where `reticulate` load hooks (normally defined via
  `setHook("reticulate::<module>::load", ...)`) would segfault if those
  hooks attempted to load the hooked module.

- `reticulate` now attempts to resolve the conda binary used to create the
  associated Conda environment in calls to `py_install()`. This should fix use
  cases where Conda environments are placed outside of the Conda installation
  itself.

- `reticulate` now sets `PYTHONPATH` before loading Python, to ensure modules
  are looked up in the same locations where a regular Python interpreter would
  find them on load. This should fix issues where `reticulate` was unable to
  bind to a Python virtual environment in some cases.
  
- `reticulate::virtualenv_create()` gains the `packages` argument, allowing one
  to choose a set of packages to be installed (via `pip install`) after the
  virtual environment has been created.

- `reticulate::virtualenv_create()` gains the `system_site_packages` argument,
  allowing one to control whether the `--system-site-packages` flag is passed
  along when creating a new virtual environment. The default value can be
  customized via the `"reticulate.virtualenv.system_site_packages"` option and
  now defaults to `FALSE` when unset.

- Fixed an issue where `reticulate::configure_environment()` would fail
  when attempting to configure an Anaconda environment. (#794)

- `reticulate` now avoids presenting a Miniconda prompt for interactive
  sessions during R session initialization.

- Fixed unsafe usages of `Rprintf()` and `REprintf()`.

- `reticulate::py_install()` better respects the `method` argument, when
  `py_install()` is called without an explicit environment name. (#777)
  
- `reticulate:::pip_freeze()` now better handles `pip` direct references.
  (#775)

- Fixed an issue where output generated from `repl_python()` would
  be buffered until the whole submitted command had completed.
  (#739, @randy3k)

- `reticulate` now explicitly qualifies symbols used from TinyThread
  with `tthread::`, to avoid issues with symbol conflicts during
  compilation. (#773)
  
- `reticulate` will now prefer an existing Miniconda installation over
  a `conda` binary on the PATH, when looking for Conda. (#790)
  
## reticulate 1.16

- TinyThread now calls `Rf_error()` rather than `std::terminate()`
  when an internal error occurs.

- Conversion of Pandas DataFrames to R no longer emits deprecation
  warnings with pandas >= 0.25.0. (#762)
  
- `reticulate` now properly handles the version strings returned by beta
  versions of `pip`. (#757)

- `conda_create()` gains the `forge` and `channel` arguments,
  analogous to those already in `conda_install()`. (#752, @jtilly)

## reticulate 1.15

- `reticulate` now ensures SciPy `csr_matrix` objects are sorted before
  attempting to convert them to their R equivalent. (#738, @paulofelipe)

- Fixed an issue where calling `input()` from Python with no prompt
  would fail. (#728)

- Lines ending with a semi-colon are no longer auto-printed in the
  `reticulate` REPL. (#717, @jsfalk)

- `reticulate` now searches for Conda binaries in /opt/anaconda and
  /opt/miniconda. (#713)

- The `conda` executable used by `reticulate` can now be configured using an R
  option. Use `options(reticulate.conda_binary = <...>)` to force `reticulate`
  to use a particular `conda` executable.

- `reticulate::use_condaenv()` better handles cases where no
  matching environment could be found. (#687)
  
- `reticulate` gains the `py_ellipsis()` function, used to access
  the Python `Ellipsis` builtin. (#700, @skeydan)

- `reticulate::configure_environment()` now only allows environment
  configuration within interactive R sessions, and ensures that the
  version of Python that has been initialized by Python is indeed
  associated with a virtual environment or Conda environment.
  Use `reticulate::configure_environment(force = TRUE)` to force
  environment configuration within non-interactive R sessions.

- `reticulate` now automatically flushes output written to Python's
  stdout / stderr, as a top-level task added by `addTaskCallback()`.
  This behavior is controlled with the `options(reticulate.autoflush)`
  option. (#685)

- `reticulate::install_miniconda()` no longer attempts to modify the
  system PATH or registry when installing Miniconda. (#681)

- `reticulate::conda_install()` gains the `channel` argument, allowing
  custom Conda channels to be used when installing Python packages.
  (#443)

- `reticulate::configure_environment()` can now be used to configure a
  non-Miniconda Python environment. (#682; @skeydan)

- Fixed an issue where matplotlib plots would be included using absolute
  paths, which fails in non-standalone documents rendered to HTML. (#669)

- Fixed an issue where `reticulate` would attempt to flush a non-existent
  stdout / stderr stream. (#584)

## reticulate 1.14

- Fixed an issue where `rmarkdown::render()` could fail when including
  matplotlib plots when `knit_root_dir` is set. (#645)

- `reticulate` now scans for Conda installations within the ~/opt folder,
  as per the updated installers distributed for macOS. (#661)

- Python classes can now be defined directly from R using the `PyClass()`
  function. (#635; @dfalbel)

- reticulate is now compatible with Python 3.9. (#630, @skeydan)

- Pandas DataFrames with a large number of columns should now be converted to
  R data.frames more quickly. (#620, @skeydan)

- Python loggers are now better behaved in the Python chunks of R Markdown
  documents. (#386)

- reticulate will now attempt to bind to `python3` rather than `python`,
  when no other version of Python has been explicitly requested by
  e.g. `use_python()`.

- reticulate now provides R hooks for Python's `input()` and `raw_input()`
  functions. It should now be possible to read user input from Python scripts
  loaded by reticulate. (#610)

- `reticulate` now more consistently normalizes the paths reported by
  `py_config()`. (#609)

- `reticulate` now provides a mechanism for allowing client packages to declare
  their Python package dependencies. Packages should declare the Python packages
  they require as part of the `Config/reticulate` field in their `DESCRIPTION` file.
  Currently, this only activated when using Miniconda; as the assumption is that
  users will otherwise prefer to manually manage their Python environments.
  Please see `vignette("python_dependencies")` for more details.

- `reticulate` will now prompt the user to create and use a
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment
  when no other suitable Python environment has already been requested. This
  should help ease some of the trouble in setting up a Python environment on
  different platforms. The installer code was contributed by @hafen, from the
  [rminiconda](https://github.com/hafen/rminiconda) package.

- Fixed an issue where `virtualenv_create(..., python = "<python>")` could
  fail to use the requested version of Python when `venv` is not installed.
  (#399)

- Fixed an issue where iterable Python objects could not be iterated with
  `iter_next()` due to a missing class. (#603)

- Fixed an issue where Conda environments could be mis-detected as
  virtual environments.

- R functions wrapping Python functions now inherit the formal arguments
  as specified by Python, making autocompletion more reliable.
  (#573, @flying-sheep)

- Fixed an issue where attempts to query Conda for environments could fail
  on Windows. (#576; #575; @dfalbel)

- Properly check for NULL keyword arguments in `call_r_function()`.
  (#562, @dfalbel)

## reticulate 1.13

- Fixed an issue where subsetting with `[.python.builtin.object` could
  fail when `convert = TRUE` is set on the associated Python object.
  (#554)

- Fixed an issue where the wrong definition of `[[.python.builtin.object`
  was being exported. (#554)

- `py_install()` now accepts `python_version`, and can be used
  if a particular version of Python is required for a Conda
  environment. (This argument is ignored for virtual environments.)
  (#549)

- Fixed an issue where reticulate could segfault in some cases
  (e.g. when using the `iterate()` function). (#551)

- It is now possible to compile `reticulate` with support for debug
  versions of Python by setting the `RETICULATE_PYTHON_DEBUG` preprocessor
  define during compilation. (#548)

- reticulate now warns if it did not honor the user's request to load a
  particular version of Python, as through e.g. `reticulate::use_python()`.
  (#545)

- `py_save_object()` and `py_load_object()` now accept `...` arguments. (#542)

- `py_install()` has been revamped, and now better detects
  available Python tooling (virtualenv vs. venv vs. Conda). (#544)

- reticulate now flushes stdout / stderr after calls to `py_run_file()` and
  `py_run_string()`.

- Python tuples are now converted recursively, in the same way that Python
  lists are. This means that the sub-elements of the tuple will be converted
  to R objects when possible. (#525, @skeydan)

- Python OrderedDict objects with non-string keys are now properly
  converted to R. (#516)

- Fixed an issue where reticulate could crash after a failed attempt
  to load NumPy. (#497, @ecoughlan)

## reticulate 1.12

- Fixed an issue where Python objects within Python lists would not be
  converted to R objects as expected.

- Fixed an issue where single-row data.frames with row names could not
  be converted. (#468)

- Fixed an issue where `reticulate` could fail to query Anaconda environment
  names with Anaconda 3.7.

- Fixed an issue where vectors of R Dates were not converted correctly. (#454)

- Fixed an issue where R Dates could not be passed to Python functions. (#458)

## reticulate 1.11.1

- Fixed a failing virtual environment test on CRAN.

## reticulate 1.11

- Fixed an issue where attempts to activate virtual environments created with
  virtualenv 16.4.1 would fail. (#437)

- Fixed an issue where conversion of Pandas Categorical variables to R objects
  would fail. (#389)

- Textual output generated when adding items to a matplotlib plot object
  are now suppressed.

- If the last statement in a Python chunk returns a matplotlib plot object,
  the plot will now be auto-shown as in other environments.

- The reticulate function help handler now returns function arguments for
  Python builtin functions.

- Top-level Python statements can now include leading indent when submitted
  with `repl_python()`.

- The current `matplotlib` figure is now cleared as each Python chunk in an
  R Markdown document is run.

- The `r` helper object (used for evaluating R code from Python) now better
  handles conversion of R functions. (#383)

- The `use_virtualenv()` function now understands how to bind to virtual
  environments created by the Python `venv` module.
  
- Reticulate better handles conversions of R lists to Python, and similarly,
  Python lists to R. We now call `r_to_py()` on each sub-element of an R list,
  and similarly, `py_to_r()` on each sub-element of a Python list.

- Reticulate now always converts R `Date` objects into Python `datetime`
  objects. Note that these conversions can be inefficient -- if you would
  prefer conversion to NumPy `datetime64` objects / arrays, you should convert
  your date to `POSIXct` first.

- Python chunks containing errors will cause execution to halt if 'error=FALSE'
  during render, conforming with the default knitr behavior for R chunks.

- The output of bare statements (e.g. `1 + 1`) is now emitted as output when using
  the reticulate Python engine.

- Remapping of Python output streams to be R can now be explicitly enabled
  by setting the environment variable `RETICULATE_REMAP_OUTPUT_STREAMS` to 1. (#335)

- Allow syntax errors in Python chunks with 'eval = FALSE' (#343)

- Avoid dropping blank lines in Python chunks (#328)

- Use "agg" matplotlib backend when running under RStudio Desktop (avoids
  crashes when attempting to generate Python plots)

- Add `as.character()` S3 method for Python bytes (defaults to converting using 
  UTF-8 encoding)
  
- Add `py_main_thread_func()` for providing R callbacks to Python libraries that may
  invoke the function on a Python background thread.

- Add `py_to_r` S3 methods for Scipy sparse matrices: CSR to dgRMatrix, COO to dgTMatrix, and for all other sparse matrices, conversion via CSC/dgCMatrix.

## reticulate 1.10

- Output is now properly displayed when using the `reticulate` REPL with
  Windows + Python 2.7.

- Address memory protection issues identified by rchk

- Make variables defined using `%as%` operator in `with()` available after 
  execution of the with block (same behavior as Python).
  
- Check for presence of "__module__" property before reading in `as_r_class()`

- Only update pip in `virtualenv_install()` when version is < 8.1

- Support converting Python `OrderedDict` to R

- Support for iterating all types of Python iterable

- Add `conda_python()` and `virtualenv_python()` functions for finding the
  python binary associated with an environment.


## reticulate 1.9 

- Detect python 3 in environments where there is no python 2 (e.g. Ubuntu 18.04)

- Always call r_to_py S3 method when converting objects from Python to R

- Handle NULL module name when determining R class for Python objects

- Convert RAW vectors to Python bytearray; Convert Python bytearray to RAW

- Use importlib for detecting modules (rather than imp) for Python >= 3.4

- Close text connection used for reading Python configuration probe


## reticulate 1.8

- `source_python()` now flushes stdout and stderr after running the associated
  Python script, to ensure that `print()`-ed output is output to the console.
  (#284)

- Fixed an issue where logical R matrices would not be converted correctly to
  their NumPy counterpart. (#280)

- Fixed an issue where Python chunks containing multiple statements on the same
  line would be evaluated and printed multiple times.

- Added `py_get_item()`, `py_set_item()`, and `py_del_item()` as lower-level
  APIs for directly accessing the items of e.g. a Python dictionary or a Pandas
  DataFrame.

- Fix issue with Pandas column names that clash with built in methods (e.g. 'pop')

- Improve default `str()` output for Python objects (print `__dict__` if available)


## reticulate 1.7

- Improved filtering of non-numeric characters in Python / NumPy versions.

- Added `py_func()` to wrap an R function in a Python function with the same signature as that of the original R function.

- Added support for conversion between `Matrix::dgCMatrix` objects in R and `Scipy` CSC matrices in Python.

- `source_python()` can now source a Python script from a URL into R environments.

- Always run `source_python()` in the main Python module.

- `py_install()` function for installing Python packages into virtualenvs and conda envs

- Automatically create conda environment for `conda_install()`

- Removed `delay_load` parameter from `import_from_path()`


## reticulate 1.6

- `repl_python()` function implementing a lightweight Python REPL in R.

- Support for converting Pandas objects (`Index`, `Series`, `DataFrame`)

- Support for converting Python `datetime` objects.

- `py_dict()` function to enable creation of dictionaries based on lists of 
  keys and values.

- Provide default base directory (e.g. '~/.virtualenvs') for environments
  specified by name in `use_virtualenv()`.
  
- Fail when environment not found with `use_condaenv(..., required = TRUE)`
  
- Ensure that `use_*` python version is satsified when using `eng_python()`

- Forward `required` argument from `use_virtualenv()` and `use_condaenv()`

- Fix leak which occurred when assigning R objects into Python containers

- Add support for Conda Forge (enabled by default) to `conda_install()`

- Added functions for managing Python virtual environments (virtualenv)


## reticulate 1.5

- Remove implicit documentation extraction for Python classes

- Add `Library\bin` to PATH on Windows to ensure Anaconda can find MKL

- New `source_python()` function for sourcing Python scripts into R 
  environments.


## reticulate 1.4

- Support for `RETICULATE_DUMP_STACK_TRACE` environment variable which can be set to
  the number of milliseconds in which to output into stderr the call stacks
  from all running threads.
  
- Provide hook to change target module when delay loading

- Scan for conda environments in system-level installations

- Support for miniconda environments

- Implement `eval`, `echo`, and `include` knitr chunk options for Python engine


## reticulate 1.3.1

- Bugfix: ensure single-line Python chunks that produce no output still 
  have source code emitted.


## reticulate 1.3

- Use existing instance of Python when reticulate is loaded within an 
  embedded Python environment (e.g. rpy2, rice, etc.)

- Force use of Python specified in PYTHON_SESSION_INITIALIZED (defined by rpy2)

- Define R_SESSION_INITIALIZED (used by rpy2)

- Force use of Python when `required = TRUE` in `use_python` functions

- Force use of Python specified by RETICULATE_PYTHON

- `dict`: Don't scan parent frame for Python objects if a single unnamed list 
  is passed.

- Wait as long as required for scheduling generator calls on the main thread

- Refine stripping of object addresses from output of `py_str()` method

- Added `py_id()` function to get globally unique ids for Python objects

- Added `py_len()` function and S3 `length()` method for Python lists (already
  had `length()` methods for dicts, tuples, and NumPy arrays).
  
- Exported `py` object (reference to Python main module)

- Added `eng_python()` (knitr engine for Python chunks)

- Improved compatibility with strings containing high unicode characters 
  when running under Python 2

- Remove `dim` methods for NumPy arrays (semantics of NumPy reshaping are
  different from R reshaping)
  
- Added `array_reshape` function for reshaping R arrays using NumPy (row-major)
  semantics.
  
- Provide mechanism for custom R wrapper objects for Python objects

- Added interface to pickle (`py_save_object()` and `py_load_object()`)

- Catch and print errors which occur in generator functions

- Write using Rprintf when providing custom Python output streams
  (enables correct handling of terminal control characters)

- Implement `isatty` when providing custom Python output streams


## reticulate 1.2

- Add `np_array` function for creating NumPy arrays and converting the data type, 
  dimensions, and in-memory ordering of existing NumPy arrays.
  
- Add `dim` and `length` functions for NumPy arrays

- Add `py_set_seed` function for setting Python and NumPy random seeds.

- Search in additional locations for Anaconda on Linux/Mac

- Improved support for UTF-8 conversions (always use UTF-8 when converting from
  Python to R)

- Ignore private ("_" prefixed) attributes of dictionaries for .DollarNames

- Provide "&#96;function&#96;" rather than "function" in completions.

- Fail gracefully if call to conda in `conda_list` results in an error

- Add `pip_ignore_installed` option to `conda_install` function.


## reticulate 1.1

- Allow `dict()` function to accept keys with mixed alpha/numeric characters

- Use `conda_list()` to discover conda environments on Windows (slower but
  much more reliable than scanning the filesystem)

- Add interface for registering F1 help handlers for Python modules

- Provide virtual/conda env hint mechanism for delay loaded imports


## reticulate 1.0

- Search WORKON_HOME (used by virtualenv_wrapper) for Python environments  

- Support `priority` field for delay loaded modules.

- Use json output from conda_list (handle spaces in path of conda env)

- Look for callable before iterable when converting Python objects to R

- Correct propagation of errors in R functions called from Python

- Support for generators (creating Python iterators from R functions)

- Changed default `completed` value for `iter_next()` to `NULL` (was `NA`)

- Support for converting 16-bit floats (NPY_HALF) to R

- Don't throw error when probing Python <= 2.6 

- Copy Python dictionary before converting to R named list (fixes issue
  with dictionaries that are mutated during iteration, e.g. sys.modules)
  
- Ensure that existing warning filters aren't reset by py_suppress_warnings


## reticulate 0.9

- Detect older versions of Anaconda during registry scanning.

- Don't probe python versions on windows when no executable is found

- Poll for interrupts every 500ms rather than 100ms

- Provide sys.stdout and sys.stderr when they are None (e.g. in R GUI)

- Add Scripts directory to PATH on Windows

- Add iter_next function for element-by-element access to iterators

- Eliminate special print method for iterators/generators

- Added `py_help()` function for printing documentation on Python objects

- Added `conda_version()` function.

- Search `dict()` parent frames for symbols; only use symbols which inherit
  from python.builtin.object as keys.


## reticulate 0.8

- Add `import_from_path()` function for importing Python modules from 
  the filesystem.

- Add `py_discover_config()` function to determine which versions of Python 
  will be discovered and which one will be used by reticulate.

- Add `py_function_docs()` amd `py_function_wrapper()` utility functions for 
  scaffolding R wrappers for Python functions.

- Add `py_last_error()` function for retreiving last Python error.

- Convert 0-dimension NumPy arrays (scalars) to single element R vectors 

- Convert "callable" Python objects to R functions

- Automatically add Python bin directory to system PATH for consistent
  version usage in reticulate and calls to system
  
- Added `length()` method for tuple objects

- Enable specification of `__name__` for R functions converted to
  Python functions.
  
- Give priority to the first registered delay load module (previously 
  the last registered module was given priority)
  
- Add additional safety checks to detect use of NULL xptr objects 
  (i.e. objects from a previous session). This should mean that S3
  methods no longer need to check whether they are handling an xptr.
  
- Added `py_eval()` function for evaluating simple Python statements.

- Add `local` option to `py_run_string()` and `py_run_file()`. Modify
  behavior to return local execution dictionary (rather than a reference
  to the main module).
  
- Use `PyImport_Import` rather than `PyImport_ImportModule` for `import()`

- Added ability to customize mapping of Python classes to R classes via
  the `as` argument to `import()` and the `register_class_filter()` function
  
- Added separate `on_load` and `on_error` functions for `delay_load`

- Scan customary root directories for virtualenv installations

- Allow calling `__getitem__` via `[[` operator (zero-based to match 
  Python style indexing)
  
- Added `conda_*` family of functions for using conda utilities from
  within R.
  
- Implement comparison operators (e.g. `==`, `>=`, etc.) for Python objects 

- Implement `names()` generic for Python objects

- Improve performance for marshalling of large Python dictionaries and 
  iterators that return large numbers of items.
  
- Implement `str` methods for Python List, Dict, and Tuple (to prevent
  printing of very large collections via default `str` method)

- Use `grepl()` rather than `endsWith()` for compatibility with R <= 3.2

- Use `inspect.getmro` rather than `__bases__` for enumerating the base classes
  of Python objects.

- Fix `PROTECT`/`UNPROTECT` issue detected by CRAN 

- Correct converstion of strings with Unicode characters on Windows
  
- Fix incompatibility with system-wide Python installations on Windows

- Fix issue with Python dictionary keys that shared names with  
  primitive R functions (don't check environment inheritance chain
  when looking for dictionary key objects by name).
  
- Propagate `convert` parameter for modules with `delay_load`

  
## reticulate 0.7

- Initial CRAN release
