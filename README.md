R Interface to Python
================

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

Installation
------------

You can install reticulate from CRAN as follows:

``` r
install.packages("reticulate")
```

### Locating Python

If the version of Python you want to use is located on the system `PATH` then it will be automatically discovered (via `Sys.which`) and used.

Alternatively, you can use one of the following functions to specify alternate versions of Python:

| Function        | Description                                           |
|-----------------|-------------------------------------------------------|
| use\_python     | Specify the path a specific Python binary.            |
| use\_virtualenv | Specify the directory containing a Python virtualenv. |
| use\_condaenv   | Specify the name of a Conda environment.              |

For example:

``` r
library(reticulate)
use_python("/usr/local/bin/python")
use_virtualenv("~/myenv")
use_condaenv("myenv")
```

Note that the NumPy features of **reticulate** require NumPy &gt;= 1.6 so versions of Python that satisfy this requirement will be preferred over ones that don't.

Also note that the `use` functions are by default considered only hints as to where to find Python (i.e. they don't produce errors if the specified version doesn't exist). You can add the `required` parameter to ensure that the specified version of Python actually exists:

``` r
use_virtualenv("~/myenv", required = TRUE)
```

The order in which versions of Python will be discovered and used is as follows:

1.  If specified, at the locations referenced by calls to `use_python`, `use_virtualenv`, and `use_condaenv`.

2.  If specified, at the location referenced by the `RETICULATE_PYTHON` environment variable.

3.  At the location of the Python binary discovered on the system `PATH` (via the `Sys.which` function).

4.  At other customary locations for Python including `/usr/local/bin/python`, `/opt/local/bin/python`, etc.

The scanning for and binding to a version of Python typically occurs at the time of the first call to `import` within an R session. As a result, priority will be given to versions of Python that include the module specified within the call to `import` (i.e. versions that don't include it will be skipped).

You can use the `py_config` function to query for information about the specific version of Python in use as well as a list of other Python versions discovered on the system:

``` r
py_config()
```

Importing Modules
-----------------

The `import` function can be used to import any Python module. For example:

``` r
difflib <- import("difflib")
difflib$ndiff(foo, bar)

filecmp <- import("filecmp")
filecmp$cmp(dir1, dir2)
```

The `import_main` and `import_builtins` functions give you access to the main module where code is executed by default and the collection of built in Python functions. For example:

``` r
main <- import_main()

py <- import_builtins()
py$print('foo')
```

The main module is generally useful if you have executed Python code from a file or string and want to get access to it's results (see the section below for more details).

### Object Conversion

By default when Python objects are returned to R they are converted to their equivalent R types. However, if you'd rather make conversion from Python to R explicit and deal in native Python objects by default you can pass `convert = FALSE` to the `import` function. In this case Python to R conversion will be disabled for the module returned from `import`. For example:

``` r
# import numpy and speicfy no automatic Python to R conversion
np <- import("numpy", convert = FALSE)

# do some array manipulations with NumPy
a <- np$array(c(1:4))
sum <- a$cumsum()

# convert to R explicitly at the end
py_to_r(sum)
```

As illustrated above, if you need access to an R object at end of your computations you can call the `py_to_r` function explicitly.

Executing Code
--------------

You can execute Python code within the main module using the `py_run_file` and `py_run_string` functions. These functions both return a reference to the main Python module so you can access the results of their execution. For example:

``` r
py_run_file("script.py")

main <- py_run_string("x = 10")
main$x
```

Error Handling
--------------

When an error occurs within Python code it is converted to an R error message of the form `<ErrorType>: <ErrorMessage>` (e.g. "ValueError: The `kernel_size` argument must be a tuple of 2 integers. Received: (2,)"). The error is signaled using the standard R error handling mechanism.

Python stack traces for errors are captured but not printed by default. To turn on printing of stack traces you can use the `reticulate.traceback` option. For example:

``` r
options(reticulate.traceback = TRUE)
```

You can also call the `py_last_error()` function to retreive detailed information on the last Python error encountered:

``` r
reticulate::py_last_error()
```

Lists, Tuples, and Dictionaries
-------------------------------

The automatic conversion of R types to Python types works well in most cases, but occasionally you will need to be more explicit on the R side to provide Python the type it expects.

For example, if a Python API requires a list and you pass a single element R vector it will be converted to a Python scalar. To overcome this simply use the R `list` function explicitly:

``` r
foo$bar(indexes = list(42L))
```

Similarly, a Python API might require a `tuple` rather than a list. In that case you can use the `tuple` function:

``` r
tuple("a", "b", "c")
```

R named lists are converted to Python dictionaries however you can also explicitly create a Python dictionary using the `dict` function:

``` r
dict(foo = "bar", index = 42L)
```

This might be useful if you need to pass a dictionary that uses a more complex object (as opposed to a string) as it's key.

With Contexts
-------------

The R `with` generic function can be used to interact with Python context manager objects (in Python you use the `with` keyword to do the same). For example:

``` r
py <- import_builtins()
with(py$open("output.txt", "w") %as% file, {
  file$write("Hello, there!")
})
```

This example opens a file and ensures that it is automatically closed at the end of the with block. Note the use of the `%as%` operator to alias the object created by the context manager.

Iterators
---------

If a Python API returns an [iterator or generator](http://anandology.com/python-practice-book/iterators.html) you can interact with it using the `iterate` function. The `iterate` function can be used to apply an R function to each item yielded by the iterator:

``` r
iterate(iter, print)
```

If you don't pass a function to `iterate` the results will be collected into an R vector:

``` r
results <- iterate(iter)
```

Note that the `Iterators` will be drained of their values by `iterate()`:

``` r
a <- iterate(iter) # results are not empty
b <- iterate(iter) # results are empty since items have already been drained
```

Callable Objects
----------------

In addition to accessing their methods and properties, some Python objects are also callable (meaning they can be invoked with parameters just like an ordinary function). Callable Python objects are returned to R as objects rather than functions, however, you can still execute the callable function via the `$call()` method, for example:

``` r
# get a callable object
parser <- spacy$English()

# call the object as a function
parser$call(spacy)
```

Advanced Functions
------------------

There are several more advanced functions available that are useful principally when creating high level R interfaces for Python libraries.

### Python Objects

Typically interacting with Python objects from R involves using the `$` operator to access whatever properties for functions of the object you need. When using the `$`, Python objects are automatically converted to their R equivalents when possible. The following functions enable you to interact with Python objects at a lower level (e.g. no conversion to R is done unless you explicitly call the `py_to_r` function):

| Function             | Description                                                 |
|----------------------|-------------------------------------------------------------|
| py\_has\_attr        | Check if an object has a specified attribute.               |
| py\_get\_attr        | Get an attribute of a Python object.                        |
| py\_set\_attr        | Set an attribute of a Python object.                        |
| py\_list\_attributes | List all attributes of a Python object.                     |
| py\_call             | Call a Python callable object with the specified arguments. |
| py\_to\_r            | Convert a Python object to it's R equivalent                |
| r\_to\_py            | Convert an R object to it's Python equivalent               |

### Configuration

The following functions enable you to query for information about the Python configuration available on the current system.

<table style="width:100%;">
<colgroup>
<col width="20%" />
<col width="79%" />
</colgroup>
<thead>
<tr class="header">
<th>Function</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>py_available</td>
<td>Check whether a Python interface is available on this system.</td>
</tr>
<tr class="even">
<td>py_numpy_available</td>
<td>Check whether the R interface to NumPy is available (requires NumPy &gt;= 1.6)</td>
</tr>
<tr class="odd">
<td>py_module_available</td>
<td>Check whether a Python module is available on this system.</td>
</tr>
<tr class="even">
<td>py_config</td>
<td>Get information on the location and version of Python in use.</td>
</tr>
</tbody>
</table>

### Output Control

These functions enable you to capture or suppress output from Python:

<table style="width:100%;">
<colgroup>
<col width="20%" />
<col width="79%" />
</colgroup>
<thead>
<tr class="header">
<th>Function</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>py_capture_output</td>
<td>Capture Python output for the specified expression and return it as an R character vector.</td>
</tr>
<tr class="even">
<td>py_suppress_warnings</td>
<td>Execute the specified expression, suppressing the display Python warnings.</td>
</tr>
</tbody>
</table>

### Miscellaneous

The functions provide miscellaneous other lower-level capabilities:

<table style="width:100%;">
<colgroup>
<col width="20%" />
<col width="79%" />
</colgroup>
<thead>
<tr class="header">
<th>Function</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>py_unicode</td>
<td>Convert a string to a Python unicode object.</td>
</tr>
<tr class="even">
<td>py_str</td>
<td>Get the string representation of Python object.</td>
</tr>
<tr class="odd">
<td>py_is_null_xptr</td>
<td>Check whether a Python object is a null externalptr.</td>
</tr>
<tr class="even">
<td>py_validate_xptr</td>
<td>Check whether a Python object is a null externalptr and throw an error if it is.</td>
</tr>
</tbody>
</table>

Using in a Package
------------------

### Checking and Testing on CRAN

If you use **reticulate** in another R package you need to account for the fact that when your package is submitted to CRAN, the CRAN test servers may not have Python, NumPy, or whatever other Python modules you are wrapping in your package. If you don't do this then your package may fail to load and/or pass it's tests when run on CRAN.

There are two things you should do to ensure your package is well behaved on CRAN:

1.  When importing Python modules for use inside your package you should use the `delay_load` option to ensure that the module (and Python) is loaded only on it's first use. For example:

    ``` r
    # python 'foo' module I want to use in my package
    foo <- NULL

    .onLoad <- function(libname, pkgname) {
      # delay load foo module (will only be loaded when accessed via $)
      foo <<- import("foo", delay_load = TRUE)
    }
    ```

2.  When writing tests, check to see if your module is available and if it isn't then skip the test. For example, if you are using the **testthat** package, you might do this:

    ``` r
    # helper function to skip tests if we don't have the 'foo' module
    skip_if_no_foo <- function() {
      have_foo <- py_module_available("foo")
      if (!have_foo)
        skip("foo not available for testing")
    }

    # then call this function from all of your tests
    test_that("Things work as expected", {
      skip_if_no_foo()
      # test code here...
    })
    ```

### S3 Methods

Python objects exposed by **reticulate** carry their Python classes into R, so it's possible to write S3 methods to customize e.g. the `str` or `print` behavior for a given class (note that it's not typically necessary that you do this since the default `str` and `print` methods call `PyObject_Str`, which typically provides an acceptable default behavior).

If you do decide to implement custom S3 methods for a Python class it's important to keep in mind that when an R session ends the connection to Python objects is lost, so when the .RData saved from one R session is restored in a subsequent R session the Python objects are effectively lost (technically they become `NULL` R `externalptr` objects).

This means that you should always use the `py_is_null_xptr` function on objects before interacting with them in an S3 method. For example:

``` r
#' @export
summary.MyPythonClass <- function(object, ...) {
  if (py_is_null_xptr(object))
    stop("Object is NULL")
  else
    # interact with the object to generate the summary
}
```

There are a couple of shortcut methods you can use to make this more straightforward. The `py_validate_xptr` function will do the check and automatically throw an error if it fails. So the example above could be re-written as:

``` r
#' @export
summary.MyPythonClass <- function(object, ...) {
  py_validate_xptr(object)
  # interact with the object to generate the summary
}
```

Finally, the **reticulate** package exports a `py_str` generic method which is called from the `str` method only after doing appropriate validation (if the object is NULL then `<pointer: 0x0>` is returned). You can implement the `py_str` method as follows:

``` r
#' @importFrom reticulate py_str
#' @export 
py_str.MyPythonClass <- function(object, ...) {
  # interact with the object to generate the string
}
```

So in short, to provide custom `str` and `print` methods, just implement `py_str`. For other S3 methods be sure to call either `py_validate_xptr` or `py_is_null_xptr` before interacting with the object.
