# Register a help handler for a root Python module

Register a help handler for a root Python module

## Usage

``` r
register_module_help_handler(module, handler)
```

## Arguments

- module:

  Name of a root Python module

- handler:

  Handler function: `function(name, subtopic = NULL)`. The name will be
  the fully qualified name of a Python object (module, function, or
  class). The `subtopic` is optional and is currently used only for
  methods within classes.

## Details

The help handler is passed a fully qualified module, class, or function
name (and optional method for classes). It should return a URL for a
help page (or `""` if no help page is available).
