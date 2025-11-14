# Check if a Python object is a null externalptr

Check if a Python object is a null externalptr

## Usage

``` r
py_is_null_xptr(x)

py_validate_xptr(x)
```

## Arguments

- x:

  Python object

## Value

Logical indicating whether the object is a null externalptr

## Details

When Python objects are serialized within a persisted R environment
(e.g. .RData file) they are deserialized into null externalptr objects
(since the Python session they were originally connected to no longer
exists). This function allows you to safely check whether whether a
Python object is a null externalptr.

The `py_validate` function is a convenience function which calls
`py_is_null_xptr` and throws an error in the case that the xptr is
`NULL`.
