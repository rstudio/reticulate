# Import a Python module

Import the specified Python module, making it available for use from R.

## Usage

``` r
import(module, as = NULL, convert = TRUE, delay_load = FALSE)

import_main(convert = TRUE, delay_load = FALSE)

import_builtins(convert = TRUE, delay_load = FALSE)

import_from_path(module, path = ".", convert = TRUE, delay_load = FALSE)
```

## Arguments

- module:

  The name of the Python module.

- as:

  An alias for module name (affects names of R classes). Note that this
  is an advanced parameter that should generally only be used in package
  development (since it affects the S3 name of the imported class and
  can therefore interfere with S3 method dispatching).

- convert:

  Boolean; should Python objects be automatically converted to their R
  equivalent? If set to `FALSE`, you can still manually convert Python
  objects to R via the
  [`py_to_r()`](https://rstudio.github.io/reticulate/dev/reference/r-py-conversion.md)
  function.

- delay_load:

  Boolean; delay loading the module until it is first used? When
  `FALSE`, the module will be loaded immediately. See **Delay Load** for
  advanced usages.

- path:

  The path from which the module should be imported.

## Value

An R object wrapping a Python module. Module attributes can be accessed
via the `$` operator, or via
[`py_get_attr()`](https://rstudio.github.io/reticulate/dev/reference/py_get_attr.md).

## Python Built-ins

Python's built-in functions (e.g. `len()`) can be accessed via Python's
built-in module. Because the name of this module has changed between
Python 2 and Python 3, we provide the function `import_builtins()` to
abstract over that name change.

## Delay Load

The `delay_load` parameter accepts a variety of inputs. If you just need
to ensure your module is lazy-loaded (e.g. because you are a package
author and want to avoid initializing Python before the user has
explicitly requested it), then passing `TRUE` is normally the right
choice.

You can also provide a named list: `"before_load"`, `"on_load"` and
`"on_error"` can be functions , which act as callbacks to be run when
the module is later loaded. `"environment"` can be a character vector of
preferred python environment names to search for and use. For example:

    delay_load = list(

      # run before the module is loaded
      before_load = function() { ... }

      # run immediately after the module is loaded
      on_load = function() { ... }

      # run if an error occurs during module import
      on_error = function(error) { ... }

      environment = c("r-preferred-venv1", "r-preferred-venv2")
    )

Alternatively, if you supply only a single function, that will be
treated as an `on_load` handler.

## Import from Path

`import_from_path()` can be used in you need to import a module from an
arbitrary filesystem path. This is most commonly used when importing
modules bundled with an R package – for example:

    path <- system.file("python", package = <package>)
    reticulate::import_from_path(<module>, path = path, delay_load = TRUE)

## Examples

``` r
if (FALSE) { # \dontrun{
main <- import_main()
sys <- import("sys")
} # }
```
