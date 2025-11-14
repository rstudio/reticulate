# Package index

## Python Requirements

- [`py_require()`](https://rstudio.github.io/reticulate/reference/py_require.md)
  : Declare Python Requirements
- [`py_write_requirements()`](https://rstudio.github.io/reticulate/reference/py_requirements_files.md)
  [`py_read_requirements()`](https://rstudio.github.io/reticulate/reference/py_requirements_files.md)
  : Write and read Python requirements files

## Python Execution

- [`import()`](https://rstudio.github.io/reticulate/reference/import.md)
  [`import_main()`](https://rstudio.github.io/reticulate/reference/import.md)
  [`import_builtins()`](https://rstudio.github.io/reticulate/reference/import.md)
  [`import_from_path()`](https://rstudio.github.io/reticulate/reference/import.md)
  : Import a Python module
- [`source_python()`](https://rstudio.github.io/reticulate/reference/source_python.md)
  : Read and evaluate a Python script
- [`repl_python()`](https://rstudio.github.io/reticulate/reference/repl_python.md)
  : Run a Python REPL
- [`eng_python()`](https://rstudio.github.io/reticulate/reference/eng_python.md)
  : A reticulate Engine for Knitr
- [`py_run_string()`](https://rstudio.github.io/reticulate/reference/py_run.md)
  [`py_run_file()`](https://rstudio.github.io/reticulate/reference/py_run.md)
  : Run Python code
- [`py_eval()`](https://rstudio.github.io/reticulate/reference/py_eval.md)
  : Evaluate a Python Expression
- [`py`](https://rstudio.github.io/reticulate/reference/py.md) :
  Interact with the Python Main Module

## Python Types

- [`dict()`](https://rstudio.github.io/reticulate/reference/dict.md)
  [`py_dict()`](https://rstudio.github.io/reticulate/reference/dict.md)
  : Create Python dictionary
- [`tuple()`](https://rstudio.github.io/reticulate/reference/tuple.md) :
  Create Python tuple
- [`as_iterator()`](https://rstudio.github.io/reticulate/reference/iterate.md)
  [`iterate()`](https://rstudio.github.io/reticulate/reference/iterate.md)
  [`iter_next()`](https://rstudio.github.io/reticulate/reference/iterate.md)
  : Traverse a Python iterator or generator
- [`py_iterator()`](https://rstudio.github.io/reticulate/reference/py_iterator.md)
  : Create a Python iterator from an R function
- [`with(`*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/with.python.builtin.object.md)
  : Evaluate an expression within a context.

## Python Configuration

- [`install_python()`](https://rstudio.github.io/reticulate/reference/install_python.md)
  : Install Python
- [`py_config()`](https://rstudio.github.io/reticulate/reference/py_config.md)
  : Python configuration
- [`py_discover_config()`](https://rstudio.github.io/reticulate/reference/py_discover_config.md)
  : Discover the version of Python to use with reticulate.
- [`py_available()`](https://rstudio.github.io/reticulate/reference/py_available.md)
  [`py_numpy_available()`](https://rstudio.github.io/reticulate/reference/py_available.md)
  : Check if Python is available on this system
- [`py_module_available()`](https://rstudio.github.io/reticulate/reference/py_module_available.md)
  : Check if a Python module is available on this system.
- [`use_python()`](https://rstudio.github.io/reticulate/reference/use_python.md)
  [`use_python_version()`](https://rstudio.github.io/reticulate/reference/use_python.md)
  [`use_virtualenv()`](https://rstudio.github.io/reticulate/reference/use_python.md)
  [`use_condaenv()`](https://rstudio.github.io/reticulate/reference/use_python.md)
  [`use_miniconda()`](https://rstudio.github.io/reticulate/reference/use_python.md)
  : Use Python
- [`py_exe()`](https://rstudio.github.io/reticulate/reference/py_exe.md)
  : Python executable
- [`py_version()`](https://rstudio.github.io/reticulate/reference/py_version.md)
  : Python version

## Python Output

- [`py_capture_output()`](https://rstudio.github.io/reticulate/reference/py_capture_output.md)
  : Capture and return Python output
- [`py_suppress_warnings()`](https://rstudio.github.io/reticulate/reference/py_suppress_warnings.md)
  : Suppress Python warnings for an expression

## Arrays

- [`np_array()`](https://rstudio.github.io/reticulate/reference/np_array.md)
  : NumPy array
- [`array_reshape()`](https://rstudio.github.io/reticulate/reference/array_reshape.md)
  : Reshape an Array

## Persistence

- [`py_save_object()`](https://rstudio.github.io/reticulate/reference/py_save_object.md)
  [`py_load_object()`](https://rstudio.github.io/reticulate/reference/py_save_object.md)
  : Save and Load Python Objects

## Low-Level Interface

- [`py_has_attr()`](https://rstudio.github.io/reticulate/reference/py_has_attr.md)
  : Check if a Python object has an attribute

- [`py_get_attr()`](https://rstudio.github.io/reticulate/reference/py_get_attr.md)
  : Get an attribute of a Python object

- [`py_set_attr()`](https://rstudio.github.io/reticulate/reference/py_set_attr.md)
  : Set an attribute of a Python object

- [`py_del_attr()`](https://rstudio.github.io/reticulate/reference/py_del_attr.md)
  : Delete an attribute of a Python object

- [`py_list_attributes()`](https://rstudio.github.io/reticulate/reference/py_list_attributes.md)
  : List all attributes of a Python object

- [`py_get_item()`](https://rstudio.github.io/reticulate/reference/py_get_item.md)
  [`py_set_item()`](https://rstudio.github.io/reticulate/reference/py_get_item.md)
  [`py_del_item()`](https://rstudio.github.io/reticulate/reference/py_get_item.md)
  [`` `[`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/py_get_item.md)
  [`` `[<-`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/py_get_item.md)
  : Get/Set/Delete an item from a Python object

- [`py_call()`](https://rstudio.github.io/reticulate/reference/py_call.md)
  : Call a Python callable object

- [`r_to_py()`](https://rstudio.github.io/reticulate/reference/r-py-conversion.md)
  [`py_to_r()`](https://rstudio.github.io/reticulate/reference/r-py-conversion.md)
  : Convert between Python and R objects

- [`as.character(`*`<python.builtin.bytes>`*`)`](https://rstudio.github.io/reticulate/reference/as.character.python.builtin.bytes.md)
  [`as.raw(`*`<python.builtin.bytes>`*`)`](https://rstudio.github.io/reticulate/reference/as.character.python.builtin.bytes.md)
  : Convert Python bytes to an R character or raw vector

- [`as.character(`*`<python.builtin.str>`*`)`](https://rstudio.github.io/reticulate/reference/as.character.python.builtin.str.md)
  : Convert a Python string to an R Character Vector

- [`py_is_null_xptr()`](https://rstudio.github.io/reticulate/reference/py_is_null_xptr.md)
  [`py_validate_xptr()`](https://rstudio.github.io/reticulate/reference/py_is_null_xptr.md)
  : Check if a Python object is a null externalptr

- [`py_id()`](https://rstudio.github.io/reticulate/reference/py_id.md) :
  Unique identifer for Python object

- [`py_len()`](https://rstudio.github.io/reticulate/reference/py_len.md)
  : Length of Python object

- [`py_bool()`](https://rstudio.github.io/reticulate/reference/py_bool.md)
  : Python Truthiness

- [`py_repr()`](https://rstudio.github.io/reticulate/reference/py_str.md)
  [`py_str()`](https://rstudio.github.io/reticulate/reference/py_str.md)
  : String representation of a python object.

- [`py_unicode()`](https://rstudio.github.io/reticulate/reference/py_unicode.md)
  : Convert to Python Unicode Object

- [`py_set_seed()`](https://rstudio.github.io/reticulate/reference/py_set_seed.md)
  : Set Python and NumPy random seeds

- [`py_clear_last_error()`](https://rstudio.github.io/reticulate/reference/py_last_error.md)
  [`py_last_error()`](https://rstudio.github.io/reticulate/reference/py_last_error.md)
  : Get or (re)set the last Python error encountered.

- [`py_help()`](https://rstudio.github.io/reticulate/reference/py_help.md)
  : Documentation for Python Objects

- [`py_func()`](https://rstudio.github.io/reticulate/reference/py_func.md)
  : Wrap an R function in a Python function with the same signature.

- [`py_main_thread_func()`](https://rstudio.github.io/reticulate/reference/py_main_thread_func.md)
  :

  [Deprecated](https://rdrr.io/r/base/Deprecated.html) Create a Python
  function that will always be called on the main thread

- [`py_ellipsis()`](https://rstudio.github.io/reticulate/reference/py_ellipsis.md)
  : The builtin constant Ellipsis

- [`py_none()`](https://rstudio.github.io/reticulate/reference/py_none.md)
  : The Python None object

- [`PyClass()`](https://rstudio.github.io/reticulate/reference/PyClass.md)
  : Create a python class

- [`py_function_custom_scaffold()`](https://rstudio.github.io/reticulate/reference/py_function_custom_scaffold.md)
  : Custom Scaffolding of R Wrappers for Python Functions

- [`nameOfClass(`*`<python.builtin.type>`*`)`](https://rstudio.github.io/reticulate/reference/nameOfClass.python.builtin.type.md)
  :

  [`nameOfClass()`](https://rdrr.io/r/base/class.html) for Python
  objects

- [`` `==`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `!=`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `<`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `>`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `>=`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `<=`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `+`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `-`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `*`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `/`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `%/%`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `%%`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `^`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `&`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `|`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `!`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  [`` `%*%`( ``*`<python.builtin.object>`*`)`](https://rstudio.github.io/reticulate/reference/Ops-python-methods.md)
  : S3 Ops Methods for Python Objects

## Package Installation

- [`py_install()`](https://rstudio.github.io/reticulate/reference/py_install.md)
  : Install Python packages
- [`py_list_packages()`](https://rstudio.github.io/reticulate/reference/py_list_packages.md)
  : List installed Python packages
- [`virtualenv_create()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.md)
  [`virtualenv_install()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.md)
  [`virtualenv_remove()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.md)
  [`virtualenv_list()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.md)
  [`virtualenv_root()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.md)
  [`virtualenv_python()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.md)
  [`virtualenv_exists()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.md)
  [`virtualenv_starter()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.md)
  : Interface to Python Virtual Environments
- [`configure_environment()`](https://rstudio.github.io/reticulate/reference/configure_environment.md)
  : Configure a Python Environment
- [`conda_list()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_create()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_clone()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_export()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_remove()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_install()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_binary()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_exe()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_version()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_update()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_python()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`conda_search()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  [`condaenv_exists()`](https://rstudio.github.io/reticulate/reference/conda-tools.md)
  : Conda Tools
- [`conda_run2()`](https://rstudio.github.io/reticulate/reference/conda_run2.md)
  : Run a command in a conda environment
- [`uv_run_tool()`](https://rstudio.github.io/reticulate/reference/uv_run_tool.md)
  : uv run tool

## Miniconda

- [`install_miniconda()`](https://rstudio.github.io/reticulate/reference/install_miniconda.md)
  : Install Miniconda
- [`miniconda_uninstall()`](https://rstudio.github.io/reticulate/reference/miniconda_uninstall.md)
  : Remove Miniconda
- [`miniconda_path()`](https://rstudio.github.io/reticulate/reference/miniconda_path.md)
  : Path to Miniconda
- [`miniconda_update()`](https://rstudio.github.io/reticulate/reference/miniconda_update.md)
  : Update Miniconda
