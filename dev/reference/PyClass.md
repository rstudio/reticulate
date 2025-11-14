# Create a python class

Create a python class

## Usage

``` r
PyClass(classname, defs = list(), inherit = NULL)
```

## Arguments

- classname:

  Name of the class. The class name is useful for S3 method dispatch.

- defs:

  A named list of class definitions - functions, attributes, etc.

- inherit:

  A list of Python class objects. Usually these objects have the
  `python.builtin.type` S3 class.

## Examples

``` r
if (FALSE) { # \dontrun{
Hi <- PyClass("Hi", list(
  name = NULL,
  `__init__` = function(self, name) {
    self$name <- name
    NULL
  },
  say_hi = function(self) {
    paste0("Hi ", self$name)
  }
))

a <- Hi("World")
} # }
```
