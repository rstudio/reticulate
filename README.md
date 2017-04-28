R Interface to Python
================

[![Travis-CI Build Status](https://travis-ci.org/rstudio/reticulate.svg?branch=master)](https://travis-ci.org/rstudio/reticulate) [![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/reticulate)](https://cran.r-project.org/package=reticulate)

Overview
--------

The **reticulate** package provides an R interface to Python modules, classes, and functions. For example, this code imports the Python `os` module and calls some functions within it:

``` r
library(reticulate)
os <- import("os")
os$chdir("tests")
os$getcwd()
```

Functions and other data within Python modules and classes can be accessed via the `$` operator (analogous to the way you would interact with an R list, environment, or reference class).

When calling into Python, R data types are automatically converted to their equivalent Python types. When values are returned from Python to R they are converted back to R types. Types are converted as follows:

| R                      | Python            | Examples                                    |
|------------------------|-------------------|---------------------------------------------|
| Single-element vector  | Scalar            | `1`, `1L`, `TRUE`, `"foo"`                  |
| Multi-element vector   | List              | `c(1.0, 2.0, 3.0)`, `c(1L, 2L, 3L)`         |
| List of multiple types | Tuple             | `list(1L, TRUE, "foo")`                     |
| Named list             | Dict              | `list(a = 1L, b = 2.0)`, `dict(x = x_data)` |
| Matrix/Array           | NumPy ndarray     | `matrix(c(1,2,3,4), nrow = 2, ncol = 2)`    |
| Function               | Python function   | `function(x) x + 1`                         |
| NULL, TRUE, FALSE      | None, True, False | `NULL`, `TRUE`, `FALSE`                     |

If a Python object of a custom class is returned then an R reference to that object is returned. You can call methods and access properties of the object just as if it was an instance of an R reference class.

The **reticulate** package is compatible with all versions of Python &gt;= 2.7. Integration with NumPy is optional and requires NumPy &gt;= 1.6.

Getting Started
------------

You can install reticulate from CRAN as follows:

``` r
install.packages("reticulate")
```

Then check out the [Introduction to reticulate](https://rstudio.github.io/reticulate/articles/introduction.html) article to learn more about using the package.


