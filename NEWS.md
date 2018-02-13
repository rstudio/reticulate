
## reticulate 1.5 (development)

Install the development version with: `install_github("rstudio/reticulate")`

- Remove implicit documentation extraction for Python classes

- Add `Library\bin` to PATH on Windows to ensure Anaconda can find MKL

- New `source_python()` function for sourcing Python scripts into R 
  environments.


## reticulate 1.4 (CRAN)

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

