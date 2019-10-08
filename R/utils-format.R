
sprintf <- function(fmt, ...) {
  
  dots <- eval(substitute(alist(...)))
  if (length(dots) == 0)
    return(fmt)
  
  base::sprintf(fmt, ...)
  
}

stopf <- function(fmt, ..., call. = FALSE) {
  stop(sprintf(fmt, ...), call. = call.)
}

warningf <- function(fmt, ..., call. = FALSE, immediate. = FALSE) {
  warning(sprintf(fmt, ...), call. = call., immediate. = immediate.)
}

messagef <- function(fmt, ..., appendLF = TRUE) {
  message(sprintf(fmt, ...), appendLF = appendLF)
}
