# Reshape an Array

Reshape (reindex) a multi-dimensional array, using row-major (C-style)
reshaping semantics by default.

## Usage

``` r
array_reshape(x, dim, order = c("C", "F"))
```

## Arguments

- x:

  An array

- dim:

  The new dimensions to be set on the array.

- order:

  The order in which elements of `x` should be read during the
  rearrangement. `"C"` means elements should be read in row-major order,
  with the last index changing fastest; `"F"` means elements should be
  read in column-major order, with the first index changing fastest.

## Details

This function differs from e.g. `dim(x) <- dim` in a very important way:
by default, `array_reshape()` will fill the new dimensions in row-major
(`C`-style) ordering, while [`dim<-()`](https://rdrr.io/r/base/dim.html)
will fill new dimensions in column-major (`F`ortran-style) ordering.
This is done to be consistent with libraries like NumPy, Keras, and
TensorFlow, which default to this sort of ordering when reshaping
arrays. See the examples for why this difference may be important.

## Examples

``` r
if (FALSE) { # \dontrun{
# let's construct a 2x2 array from a vector of 4 elements
x <- 1:4

# rearrange will fill the array row-wise
array_reshape(x, c(2, 2))
#      [,1] [,2]
# [1,]    1    2
# [2,]    3    4
# setting the dimensions 'fills' the array col-wise
dim(x) <- c(2, 2)
x
#      [,1] [,2]
# [1,]    1    3
# [2,]    2    4
} # }
```
