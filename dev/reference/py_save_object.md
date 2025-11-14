# Save and Load Python Objects

Save and load Python objects.

## Usage

``` r
py_save_object(object, filename, pickle = "pickle", ...)

py_load_object(filename, pickle = "pickle", ..., convert = TRUE)
```

## Arguments

- object:

  A Python object.

- filename:

  The output file name. Note that the file extension `.pickle` is
  considered the "standard" extension for serialized Python objects as
  created by the `pickle` module.

- pickle:

  The "pickle" implementation to use. Defaults to `"pickle`", but other
  compatible Python "pickle" implementations (e.g. `"cPickle"`) could be
  used as well.

- ...:

  Optional arguments, to be passed to the `pickle` module's
  [`dump()`](https://rdrr.io/r/base/dump.html) and
  [`load()`](https://rdrr.io/r/base/load.html) functions.

- convert:

  Bool. Whether the loaded pickle object should be converted to an R
  object.

## Details

Python objects are serialized using the `pickle` module – see
<https://docs.python.org/3/library/pickle.html> for more details.
