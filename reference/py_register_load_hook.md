# Register a Python module load hook

Register an R function to be called when a Python module is first loaded
in the current R session. This can be used for tasks such as:

## Usage

``` r
py_register_load_hook(module, hook)
```

## Arguments

- module:

  String, the name of the Python module.

- hook:

  Function, called with no arguments. If `module` is already loaded,
  `hook()` is called immediately.

## Value

`NULL` invisibly. Called for its side effect.

## Details

- Delayed registration of S3 methods to accommodate different versions
  of a Python module.

- Configuring module-specific logging streams.
