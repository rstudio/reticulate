# Traverse a Python iterator or generator

Traverse a Python iterator or generator

## Usage

``` r
as_iterator(x)

iterate(it, f = base::identity, simplify = TRUE)

iter_next(it, completed = NULL)
```

## Arguments

- x:

  Python iterator or iterable

- it:

  Python iterator or generator

- f:

  Function to apply to each item. By default applies the `identity`
  function which just reflects back the value of the item.

- simplify:

  Should the result be simplified to a vector if possible?

- completed:

  Sentinel value to return from `iter_next()` if the iteration completes
  (defaults to `NULL` but can be any R value you specify).

## Value

For `iterate()`, A list or vector containing the results of calling `f`
on each item in `x` (invisibly); For `iter_next()`, the next value in
the iteration (or the sentinel `completed` value if the iteration is
complete).

## Details

Simplification is only attempted all elements are length 1 vectors of
type "character", "complex", "double", "integer", or "logical".
