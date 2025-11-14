# Get/Set/Delete an item from a Python object

Access an item from a Python object, similar to how `x[key]` might be
used in Python code to access an item indexed by `key` on an object `x`.
The object's `__getitem__()` `__setitem__()` or `__delitem__()` method
will be called.

## Usage

``` r
py_get_item(x, key, silent = FALSE)

py_set_item(x, key, value)

py_del_item(x, key)

# S3 method for class 'python.builtin.object'
x[...]

# S3 method for class 'python.builtin.object'
x[...] <- value
```

## Arguments

- x:

  A Python object.

- key, ...:

  The key used for item lookup.

- silent:

  Boolean; when `TRUE`, attempts to access missing items will return
  `NULL` rather than throw an error.

- value:

  The item value to set. Assigning `value` of `NULL` calls
  `py_del_item()` and is equivalent to the python expression
  `del x[key]`. To set an item value of `None`, you can call
  `py_set_item()` directly, or call `x[key] <- py_none()`

## Value

For `py_get_item()` and `[`, the return value from the `x.__getitem__()`
method. For `py_set_item()`, `py_del_item()` and `[<-`, the mutate
object `x` is returned.

## Note

The `py_get_item()` always returns an unconverted python object, while
`[` will automatically attempt to convert the object if `x` was created
with `convert = TRUE`.

## Examples

``` r
if (FALSE) { # \dontrun{

## get/set/del item from Python dict
x <- r_to_py(list(abc = "xyz"))

#'   # R expression    | Python expression
# -------------------- | -----------------
 x["abc"]              # x["abc"]
 x["abc"] <- "123"     # x["abc"] = "123"
 x["abc"] <- NULL      # del x["abc"]
 x["abc"] <- py_none() # x["abc"] = None

## get item from Python list
x <- r_to_py(list("a", "b", "c"))
x[0]

## slice a NumPy array
x <- np_array(array(1:64, c(4, 4, 4)))

# R expression | Python expression
# ------------ | -----------------
  x[0]         # x[0]
  x[, 0]       # x[:, 0]
  x[, , 0]     # x[:, :, 0]

  x[NA:2]      # x[:2]
  x[`:2`]      # x[:2]

  x[2:NA]      # x[2:]
  x[`2:`]      # x[2:]

  x[NA:NA:2]   # x[::2]
  x[`::2`]     # x[::2]

  x[1:3:2]     # x[1:3:2]
  x[`1:3:2`]   # x[1:3:2]

  x[.., 1]     # x[..., 1]
  x[0, .., 1]  # x[0, ..., 1]

} # }
```
