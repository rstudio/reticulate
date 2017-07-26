
# reticulate 1.1 (unreleased)



# reticulate 1.0

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


# reticulate 0.9

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


# reticulate 0.8

## Features

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


## Bug Fixes

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

  
# reticulate 0.7

- Initial CRAN release

