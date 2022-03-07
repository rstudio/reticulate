

#' Create a Python iterator from an R function
#'
#' @param fn R function with no arguments.
#' @param completed Special sentinel return value which indicates that
#'  iteration is complete (defaults to `NULL`)
#'
#' @return Python iterator which calls the R function for each iteration.
#'
#' @details Python generators are functions that implement the Python iterator
#' protocol. In Python, values are returned using the `yield` keyword. In R,
#' values are simply returned from the function.
#'
#' In Python, the `yield` keyword enables successive iterations to use the state
#' of previous iterations. In R, this can be done by returning a function that
#' mutates its enclosing environment via the `<<-` operator. For example:
#'
#' ```r
#' sequence_generator <- function(start) {
#'   value <- start
#'   function() {
#'     value <<- value + 1
#'     value
#'   }
#' }
#' ```
#' Then create an iterator using `py_iterator()`:
#' ```r
#' g <- py_iterator(sequence_generator(10))
#' ```
#'
#' @section Ending Iteration:
#'
#' In Python, returning from a function without calling `yield` indicates the
#' end of the iteration. In R however, `return` is used to yield values, so
#' the end of iteration is indicated by a special return value (`NULL` by
#' default, however this can be changed using the `completed` parameter). For
#' example:
#'
#' ```r
#' sequence_generator <-function(start) {
#'   value <- start
#'   function() {
#'     value <<- value + 1
#'     if (value < 100)
#'       value
#'     else
#'       NULL
#'   }
#' }
#'
#' @section Threading:
#'
#' Some Python APIs use generators to parallellize operations by calling the
#' generator on a background thread and then consuming its results on
#' the foreground thread. The `py_iterator()` function creates threadsafe
#' iterators by ensuring that the R function is always called on the main
#' thread (to be compatible with R's single-threaded runtime) even if the
#' generator is run on a background thread.
#'
#' @export
py_iterator <- function(fn, completed = NULL) {

  # validation
  if (!is.function(fn))
    stop("fn must be an R function")
  if (length(formals(fn)))
    stop("fn must be an R function with no arguments")

  # wrap the function in an error handler
  wrapped_fn <- function() {
    tryCatch({
      fn()
    }, error = function(e) {
      message("Error occurred in generator: ", e$message)
      completed
    })
  }

  # create the generator
  tools <- import("rpytools")
  tools$generator$RGenerator(wrapped_fn, completed)
}

