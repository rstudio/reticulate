# Unique identifer for Python object

Get a globally unique identifier for a Python object.

## Usage

``` r
py_id(object)
```

## Arguments

- object:

  Python object

## Value

Unique identifer (as string) or `NULL`

## Note

In the current implementation of CPython this is the memory address of
the object.
