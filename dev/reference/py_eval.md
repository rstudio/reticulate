# Evaluate a Python Expression

Evaluate a single Python expression, in a way analogous to the Python
[`eval()`](https://rdrr.io/r/base/eval.html) built-in function.

## Usage

``` r
py_eval(code, convert = TRUE)
```

## Arguments

- code:

  A single Python expression.

- convert:

  Boolean; automatically convert Python objects to R?

## Value

The result produced by evaluating `code`, converted to an `R` object
when `convert` is set to `TRUE`.

## Caveats

`py_eval()` only supports evaluation of 'simple' Python expressions.
Other expressions (e.g. assignments) will fail; e.g.

    > py_eval("x = 1")
    Error in py_eval_impl(code, convert) :
      SyntaxError: invalid syntax (reticulate_eval, line 1)

and this mirrors what one would see in a regular Python interpreter:

    >>> eval("x = 1")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<string>", line 1
    x = 1
    ^
      SyntaxError: invalid syntax

The
[`py_run_string()`](https://rstudio.github.io/reticulate/dev/reference/py_run.md)
method can be used if the evaluation of arbitrary Python code is
required.
