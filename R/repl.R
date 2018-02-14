#' Run a Python REPL
#' 
#' This function provides a Python REPL in the R session, which can be used
#' to interactively run Python code. All code executed within the REPL is
#' run within the Python 'main' module, and will remain accessible after the
#' REPL is detached.
#' 
#' @param quiet Boolean; print a startup banner when launching the REPL? If
#'   `FALSE`, the banner will be suppressed.
#' @param modules A set of Python modules (packages) to be imported before
#'   the REPL is launched. Defaults to the value of the `reticulate.repl.modules`
#'   option.
#' 
#' @export
python_repl <- function(
  quiet = FALSE,
  modules = getOption("reticulate.repl.modules", default = character()))
{
  # load modules (do this first so that we bind to the expected version of
  # Python if this is the first attempt to load Python)
  for (module in modules)
    import(module)
  
  codeop <- import("codeop", convert = TRUE)
  
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
  options(reticulate.repl = TRUE)
  on.exit(options(reticulate.repl = FALSE), add = TRUE)
  
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
    
    # update history file
    if (has_history) {
      write(contents, file = histfile, append = TRUE)
      utils::loadhistory(histfile)
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
