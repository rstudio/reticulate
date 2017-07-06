

#' Create a Python iterator from an R function
#'
#' @param fn R function with no arguments.
#' @param stop Special sentinel return value which indicates that 
#'  iteration is complete.
#'
#' @return Python iterator which calls the R function for each iteration.
#'
#' @details Python generators are functions that implement the Python iterator
#' protocol. In Python, values are returned using the `yield` keyword. In R,
#' values are simply returned from the function.
#'
#' In Python, the `yield` keyword enables successive iterations to use the state
#' of previous iterations. In R, this can be done by returning a function that
#' accesses it's enclosing environment via the `<<-` operator. For example:
#'
#' ```r
#' sequence_generator <-function(start) {
#'   value <- start
#'   function() {
#'     value <<- value + 1
#'     value
#'   }
#' }
#' 
#' g <- generator(sequence_generator(10))
#' ```
#' 
#' @section Threading:
#' 
#' Some Python APIs use generators to parallelize operations by calling the
#' generator on a background thread and then consuming it's results on 
#' the foreground thread. The `generator()` function creates threadsafe
#' iterators by ensuring that the R function is always called on the main
#' thread (to be compatible with R's single-threaded runtime) even if the
#' generator is run on a background thread.
#' 
#' @export
generator <- function(fn, stop = NULL) {
  
  # validation
  if (!is.function(fn)) 
    stop("fn must be an R function")
  if (length(formals(fn) != 0))
    stop("fn must be an R function with no arguments")
  
  # create the generator
  tools <- import("rpytools")
  tools$generator$RGenerator(fn, stop)
}