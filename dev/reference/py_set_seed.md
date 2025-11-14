# Set Python and NumPy random seeds

Set various random seeds required to ensure reproducible results. The
provided `seed` value will establish a new random seed for Python and
NumPy, and will also (by default) disable hash randomization.

## Usage

``` r
py_set_seed(seed, disable_hash_randomization = TRUE)
```

## Arguments

- seed:

  A single value, interpreted as an integer

- disable_hash_randomization:

  Disable hash randomization, which is another common source of variable
  results. See
  <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>

## Details

This function does not set the R random seed, for that you should call
[`set.seed()`](https://rdrr.io/r/base/Random.html).
