# Create a Python iterator from an R function

Create a Python iterator from an R function

## Usage

``` r
py_iterator(fn, completed = NULL, prefetch = 0L)
```

## Arguments

- fn:

  R function with no arguments.

- completed:

  Special sentinel return value which indicates that iteration is
  complete (defaults to `NULL`).

- prefetch:

  Number items to prefetch. Set this to a positive integer to avoid a
  deadlock in situations where the generator values are consumed by
  python background threads while the main thread is blocked.

## Value

Python iterator which calls the R function for each iteration.

## Details

Python generators are functions that implement the Python iterator
protocol. In Python, values are returned using the `yield` keyword. In
R, values are simply returned from the function.

In Python, the `yield` keyword enables successive iterations to use the
state of previous iterations. In R, this can be done by returning a
function that mutates its enclosing environment via the `<<-` operator.
For example:

    sequence_generator <- function(start) {
      value <- start
      function() {
        value <<- value + 1
        value
      }
    }

Then create an iterator using `py_iterator()`:

    g <- py_iterator(sequence_generator(10))

## Ending Iteration

In Python, returning from a function without calling `yield` indicates
the end of the iteration. In R however, `return` is used to yield
values, so the end of iteration is indicated by a special return value
(`NULL` by default, however this can be changed using the `completed`
parameter). For example:

    sequence_generator <-function(start) {
      value <- start
      function() {
        value <<- value + 1
        if (value < 100)
          value
        else
          NULL
      }
    }

## Threading

Some Python APIs use generators to parallellize operations by calling
the generator on a background thread and then consuming its results on
the foreground thread. The `py_iterator()` function creates threadsafe
iterators by ensuring that the R function is always called on the main
thread (to be compatible with R's single-threaded runtime) even if the
generator is run on a background thread.
