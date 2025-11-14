# S3 Ops Methods for Python Objects

Reticulate provides S3 Ops Group Generic Methods for Python objects. The
methods invoke the equivalent python method of the object.

## Usage

``` r
# S3 method for class 'python.builtin.object'
e1 == e2

# S3 method for class 'python.builtin.object'
e1 != e2

# S3 method for class 'python.builtin.object'
e1 < e2

# S3 method for class 'python.builtin.object'
e1 > e2

# S3 method for class 'python.builtin.object'
e1 >= e2

# S3 method for class 'python.builtin.object'
e1 <= e2

# S3 method for class 'python.builtin.object'
e1 + e2

# S3 method for class 'python.builtin.object'
e1 - e2

# S3 method for class 'python.builtin.object'
e1 * e2

# S3 method for class 'python.builtin.object'
e1/e2

# S3 method for class 'python.builtin.object'
e1%/%e2

# S3 method for class 'python.builtin.object'
e1%%e2

# S3 method for class 'python.builtin.object'
e1^e2

# S3 method for class 'python.builtin.object'
e1 & e2

# S3 method for class 'python.builtin.object'
e1 | e2

# S3 method for class 'python.builtin.object'
!e1

# S3 method for class 'python.builtin.object'
x %*% y
```

## Arguments

- e1, e2, x, y:

  A python object.

## Value

Result from evaluating the Python expression. If either of the arguments
to the operator was a Python object with `convert=FALSE`, then the
result will also be a Python object with `convert=FALSE` set. Otherwise,
the result will be converted to an R object if possible.

## Operator Mappings

|              |                   |                              |
|--------------|-------------------|------------------------------|
| R expression | Python expression | First python method invoked  |
| `x == y`     | `x == y`          | `type(x).__eq__(x, y)`       |
| `x != y`     | `x != y`          | `type(x).__ne__(x, y)`       |
| `x < y`      | `x < y`           | `type(x).__lt__(x, y)`       |
| `x > y`      | `x > y`           | `type(x).__gt__(x, y)`       |
| `x >= y`     | `x >= y`          | `type(x).__ge__(x, y)`       |
| `x <= y`     | `x <= y`          | `type(x).__le__(x, y)`       |
| `+ x `       | `+ x`             | `type(x).__pos__(x)`         |
| `- y`        | `- x`             | `type(x).__neg__(x)`         |
| `x + y`      | `x + y`           | `type(x).__add__(x, y)`      |
| `x - y`      | `x - y`           | `type(x).__sub__(x, y)`      |
| `x * y`      | `x * y`           | `type(x).__mul__(x, y)`      |
| `x / y`      | `x / y`           | `type(x).__truediv__(x, y)`  |
| `x %/% y`    | `x // y`          | `type(x).__floordiv__(x, y)` |
| `x %% y`     | `x % y`           | `type(x).__mod__(x, y)`      |
| `x ^ y`      | `x ** y`          | `type(x).__pow__(x, y)`      |
| `x & y`      | `x & y`           | `type(x).__and__(x, y)`      |
| `x | y`      | `x | y`           | `type(x).__or__(x, y)`       |
| `!x`         | `~x`              | `type(x).__not__(x)`         |
| `x %*% y`    | `x @ y`           | `type(x).__matmul__(x, y)`   |

Note: If the initial Python method invoked raises a `NotImplemented`
Exception, the Python interpreter will attempt to use the reflected
variant of the method from the second argument. The arithmetic operators
will call the equivalent double underscore (dunder) method with an "r"
prefix. For instance, when evaluating the expression `x + y`, if
`type(x).__add__(x, y)` raises a `NotImplemented` exception, then the
interpreter will attempt `type(y).__radd__(y, x)`. The comparison
operators follow a different sequence of fallbacks; refer to the Python
documentation for more details.
