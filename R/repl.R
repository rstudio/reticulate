#' Run a Python REPL
#' 
#' This function provides a Python REPL in the R session, which can be used
#' to interactively run Python code. All code executed within the REPL is
#' run within the Python 'main' module, and will remain accessible after the
#' REPL is detached.
#' 
#' @export
py_repl <- function() {
 
  codeop <- import("codeop", convert = TRUE)
  
  # buffer of pending console input (we don't send input to
  # the console until we have a complete Python statement)
  buffer <- character()
  
  # command compiler (used to check if we've received a complete piece
  # of Python input)
  compiler <- codeop$CommandCompiler()
  
  # record whether the used has requested a quit
  quit_requested <- FALSE
  
  repl <- function() {
    
    # read user input
    prompt <- if (length(buffer)) "... " else ">>> "
    contents <- readline(prompt = prompt)
    
    # special handling for e.g. 'quit', 'exit'
    if (contents %in% c("quit", "exit")) {
      quit_requested <<- TRUE
      return()
    }
    
    # update buffer
    buffer <<- c(buffer, contents)
    
    # generate code to be sent to command interpreter
    code <- paste(buffer, collapse = "\n")
    compiled <- tryCatch(compiler(code), condition = identity)
    
    # a NULL return implies that we can accept more input
    if (is.null(compiled))
      return()
    
    # otherwise, we should have received a code output object
    # so we can just run the code submitted thus far
    buffer <<- character()
    output <- tryCatch(py_run_string(code), condition = identity)
    
    # if execution failed, let the user know
    if (inherits(output, "error")) {
      error <- py_last_error()
      message(paste(error$type, error$value, sep = ": "))
    }
    
  }
  
  # notify the user we're entering a reticulate-powered Python REPL
  config <- py_config()
  fmt <- "Python %s\nReticulate %s REPL -- A Python interpreter in R."
  msg <- sprintf(fmt, config$version_string, packageVersion("reticulate"))
  message(msg)
  
  # REPL ----
  repeat {
    
    if (quit_requested)
      break
    
    repl()
  }
  
}
