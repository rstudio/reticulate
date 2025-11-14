# NumPy array

Create NumPy arrays and convert the data type and in-memory ordering of
existing NumPy arrays.

## Usage

``` r
np_array(data, dtype = NULL, order = "C")
```

## Arguments

- data:

  Vector or existing NumPy array providing data for the array

- dtype:

  Numpy data type (e.g. "float32", "float64", etc.)

- order:

  Memory ordering for array. "C" means C order, "F" means Fortran order.

## Value

A NumPy array object.
