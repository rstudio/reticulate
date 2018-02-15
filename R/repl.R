#' Run a Python REPL
#' 
#' This function provides a Python REPL in the R session, which can be used
#' to interactively run Python code. All code executed within the REPL is
#' run within the Python 'main' module, and will remain accessible after the
#' REPL is detached.
#' 
#' @param module An (optional) Python module to be imported before
#'   the REPL is launched.
#'   
#' @param quiet Boolean; print a startup banner when launching the REPL? If
#'   `FALSE`, the banner will be suppressed.
#' 
#' @export
py_repl <- function(
  module = NULL,
  quiet = getOption("reticulate.repl.quiet", default = FALSE))
{
  # load module if requested
  if (is.character(module))
    import(module)
  
  # import other required modules for the REPL
  builtins <- import_builtins(convert = FALSE)
  main <- import_main(convert = FALSE)
  codeop <- import("codeop", convert = TRUE)
  
  # grab references to the locals, globals of the main module
  locals <- py_run_string("locals()")
  globals <- py_run_string("globals()")
  
  # check to see if the current environment supports history
  has_history <- tryCatch(
    { utils::savehistory(tempfile()); TRUE },
    error = function(e) FALSE
  )
  
  if (has_history) {
    
    # if we have history, save and then restore the current R history
    utils::savehistory()
    on.exit(utils::loadhistory(), add = TRUE)
    
    # file to be used for command history during session
    histfile <- getOption("reticulate.repl.histfile")
    if (is.null(histfile))
      histfile <- file.path(tempdir(), ".reticulatehistory")
    
    # load history (create emptu file if none exists yet)
    if (!file.exists(histfile))
      file.create(histfile)
    utils::loadhistory(histfile)
    
  }
  
  # buffer of pending console input (we don't send input to
  # the console until we have a complete Python statement)
  buffer <- character()
  
  # command compiler (used to check if we've received a complete piece
  # of Python input)
  compiler <- codeop$CommandCompiler()
  
  # record whether the used has requested a quit
  quit_requested <- FALSE
  
  # inform others that the reticulate REPL is active
  options(reticulate.repl.active = TRUE)
  on.exit(options(reticulate.repl.active = FALSE), add = TRUE)
  
  # handle errors produced during REPL actions
  handle_error <- function(output) {
    failed <- inherits(output, "error")
    if (failed) {
      error <- py_last_error()
      message(paste(error$type, error$value, sep = ": "))
    }
    failed
  }
  
  repl <- function() {
    
    # read user input
    prompt <- if (length(buffer)) "... " else ">>> "
    contents <- readline(prompt = prompt)
    
    # trim whitespace for handling of special commands
    trimmed <- gsub("^\\s*|\\s*$", "", contents)
    
    # special handling for e.g. 'quit', 'exit'
    if (trimmed %in% c("quit", "exit")) {
      quit_requested <<- TRUE
      return()
    }
    
    # if the user submitted a blank line at the top level,
    # ignore it (but submit whitespace-only lines that might
    # terminate a block)
    if (length(buffer) == 0 && !nzchar(trimmed))
      return()
    
    # update history file
    if (has_history) {
      write(contents, file = histfile, append = TRUE)
      utils::loadhistory(histfile)
    }
    
    # update buffer
    buffer <<- c(buffer, contents)
    
    # generate code to be sent to command interpreter. an error generated
    # at this pooint implies something like a syntax error; we should clear
    # the command buffer at that point
    code <- paste(buffer, collapse = "\n")
    ready <- tryCatch(compiler(code), condition = identity)
    if (handle_error(ready)) {
      buffer <<- character()
      return()
    }
    
    # a NULL return implies that we can accept more input
    if (is.null(ready))
      return()
    
    # otherwise, we should have received a code output object
    # so we can just run the code submitted thus far
    buffer <<- character()
    
    # now compile and run the code. we use 'single' mode to ensure that
    # python auto-prints the statement as it is evaluated.
    compiled <- tryCatch(builtins$compile(code, '<string>', 'single'), error = identity)
    if (handle_error(compiled))
      return()
    
    output <- tryCatch(builtins$eval(compiled, locals, globals), error = identity)
    if (handle_error(output))
      return()
    
  }
  
  # notify the user we're entering the REPL (when requested)
  if (!quiet) {
    
    config <- py_config()
    
    fmt <- c(
      "Python %s",
      "Reticulate %s REPL -- A Python interpreter in R."
    )
    
    msg <- sprintf(
      paste(fmt, collapse = "\n"),
      config$version_string,
      packageVersion("reticulate")
    )
    
    message(msg)
    
  }
  
  # enter the REPL loop
  repeat {
    
    if (quit_requested)
      break
    
    repl()
  }
  
}
