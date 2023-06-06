
sprintf <- function(fmt, ...) {
  
  dots <- eval(substitute(alist(...)))
  if (length(dots) == 0)
    return(fmt)
  
  base::sprintf(fmt, ...)
  
}

stopf <- function(fmt = "", ..., call. = FALSE) {
  stop(sprintf(fmt, ...), call. = call.)
}

warningf <- function(fmt = "", ..., call. = FALSE, immediate. = FALSE) {
  warning(sprintf(fmt, ...), call. = call., immediate. = immediate.)
}

messagef <- function(fmt = "", ..., appendLF = TRUE) {
  message(sprintf(fmt, ...), appendLF = appendLF)
}

printf <- function(fmt = "", ..., file = stdout()) {
  if (!is.null(fmt))
    cat(sprintf(fmt, ...), file = file, sep = "")
}

eprintf <- function(fmt = "", ..., file = stderr()) {
  if (!is.null(fmt))
    cat(sprintf(fmt, ...), file = file, sep = "")
}

writef <- function(fmt = "", ..., con = stdout()) {
  if (!is.null(fmt))
    writeLines(sprintf(fmt, ...), con = con)
}

ewritef <- function(fmt = "", ..., con = stderr()) {
  if (!is.null(fmt))
    writeLines(sprintf(fmt, ...), con = con)
}
