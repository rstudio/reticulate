# Evaluate an expression within a context.

The `with` method for objects of type `python.builtin.object` implements
the context manager protocol used by the Python `with` statement. The
passed object must implement the [context
manager](https://docs.python.org/3/reference/datamodel.html#context-managers)
(`__enter__` and `__exit__` methods.

## Usage

``` r
# S3 method for class 'python.builtin.object'
with(data, expr, as = NULL, ...)
```

## Arguments

- data:

  Context to enter and exit

- expr:

  Expression to evaluate within the context

- as:

  Name of variable to assign context to for the duration of the
  expression's evaluation (optional).

- ...:

  Unused
