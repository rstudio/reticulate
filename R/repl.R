#' Run a Python REPL
#'
#' This function provides a Python REPL in the \R session, which can be used
#' to interactively run Python code. All code executed within the REPL is
#' run within the Python main module, and any generated Python objects will
#' persist in the Python session after the REPL is detached.
#'
#' When working with R and Python scripts interactively, one can activate
#' the Python REPL with `repl_python()`, run Python code, and later run `exit`
#' to return to the \R console.
#'
#' @examples \dontrun{
#'
#' # enter the Python REPL, create a dictionary, and exit
#' repl_python()
#' dictionary = {'alpha': 1, 'beta': 2}
#' exit
#'
#' # access the created dictionary from R
#' py$dictionary
#' # $alpha
#' # [1] 1
#' #
#' # $beta
#' # [1] 2
#'
#' }
#'
#' @param module An (optional) Python module to be imported before
#'   the REPL is launched.
#'
#' @param quiet Boolean; print a startup banner when launching the REPL? If
#'   `TRUE`, the banner will be suppressed.
#'
#' @seealso [py], for accessing objects created using the Python REPL.
#'
#' @importFrom utils packageVersion
#' @export
repl_python <- function(
  module = NULL,
  quiet = getOption("reticulate.repl.quiet", default = FALSE))
{
  # load module if requested
  if (is.character(module))
    import(module)

  # run hooks for initialize, teardown
  initialize <- getOption("reticulate.repl.initialize")
  if (is.function(initialize)) {
    initialize()
  }

  teardown <- getOption("reticulate.repl.teardown")
  if (is.function(teardown)) {
    on.exit(teardown(), add = TRUE)
  }

  # import other required modules for the REPL
  sys <- import("sys", convert = TRUE)
  codeop <- import("codeop", convert = TRUE)

  # check to see if the current environment supports history
  # (check for case where working directory not writable)
  use_history <-
    !"--vanilla" %in% commandArgs() &&
    !"--no-save" %in% commandArgs() &&
    !is.null(getwd()) &&
    tryCatch(
      { utils::savehistory(tempfile()); TRUE },
      error = function(e) FALSE
    )

  if (use_history) {

    # if we have history, save and then restore the current
    # R history
    utils::savehistory()
    on.exit(utils::loadhistory(), add = TRUE)

    # file to be used for command history during session
    histfile <- getOption("reticulate.repl.histfile")
    if (is.null(histfile))
      histfile <- file.path(tempdir(), ".reticulatehistory")

    # load history (create empty file if none exists yet)
    if (!file.exists(histfile))
      file.create(histfile)
    utils::loadhistory(histfile)

  }

  # buffer of pending console input (we don't evaluate code
  # until the user has submitted a complete Python statement)
  #
  # we return an environment of functions bound in a local environment
  # so that hook can manipulate the buffer if required
  buffer <- new_stack()

  # command compiler (used to check if we've received a complete piece
  # of Python input)
  compiler <- codeop$CommandCompiler()

  # record whether the used has requested a quit
  quit_requested <- FALSE

  # inform others that the reticulate REPL is active
  .globals$py_repl_active <- TRUE
  on.exit(.globals$py_repl_active <- FALSE, add = TRUE)

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
    
    # flush stdout, stderr on each REPL iteration
    on.exit({
      
      if (!is.null(sys$stdout) && !is.null(sys$stdout$flush))
        sys$stdout$flush()
      
      if (!is.null(sys$stderr) && !is.null(sys$stderr$flush))
        sys$stderr$flush()
      
    }, add = TRUE)

    # read user input
    prompt <- if (buffer$empty()) ">>> " else "... "
    contents <- readline(prompt = prompt)

    # NULL implies the user sent EOF -- time to leave
    if (is.null(contents)) {
      writeLines("exit", con = stdout())
      quit_requested <<- TRUE
      return()
    }

    # trim whitespace for handling of special commands
    trimmed <- gsub("^\\s*|\\s*$", "", contents)

    # run hook provided by front-end (in case special actions
    # need to be taken in response to console input)
    hook <- getOption("reticulate.repl.hook")
    if (is.function(hook)) {

      status <- tryCatch(hook(buffer, contents, trimmed), error = identity)

      # report errors to the user
      if (inherits(status, "error")) {
        message(paste("Error:", conditionMessage(status)))
        return()
      }

      # a TRUE return implies the hook handled this input
      if (isTRUE(status))
        return()
    }

    # special handling for top-level commands (when buffer is empty)
    if (buffer$empty()) {

      # handle user requests to quit
      if (trimmed %in% c("quit", "exit")) {
        quit_requested <<- TRUE
        return()
      }

      # special handling for help requests prefixed with '?'
      if (regexpr("?", trimmed, fixed = TRUE) == 1) {
        code <- sprintf("help(\"%s\")", substring(trimmed, 2))
        py_run_string(code)
        return()
      }

      # similar handling for help requests postfixed with '?'
      if (grepl("[?]\\s*$", trimmed)) {
        replaced <- sub("[?]\\s*$", "", trimmed)
        code <- sprintf("help(\"%s\")", replaced)
        py_run_string(code)
        return()
      }

      # if the user submitted a blank line at the top level,
      # ignore it (note that we intentionally submit whitespace-only
      # lines that might terminate a block)
      if (!nzchar(trimmed))
        return()

    }

    # update history file
    if (use_history)
      cat(contents, file = histfile, sep = "\n", append = TRUE)

    # trim whitespace if the buffer is empty (this effectively allows leading
    # whitespace in top-level Python commands)
    if (buffer$empty()) contents <- trimmed

    # update buffer
    previous <- buffer$data()
    buffer$push(contents)

    # generate code to be sent to command interpreter
    code <- paste(buffer$data(), collapse = "\n")
    ready <- tryCatch(compiler(code), condition = identity)

    # a NULL return implies that we can accept more input
    if (is.null(ready))
      return()

    # on error, attempt to submit the previous buffer and then handle
    # the newest line of code independently. this allows us to handle
    # python constructs such as:
    #
    #   def foo():
    #     return 42
    #   foo()
    #
    #   try:
    #     print 1
    #   except:
    #     print 2
    #   print 3
    #
    # which would otherwise fail
    if (length(previous) && inherits(ready, "error")) {

      # submit previous code
      pasted <- paste(previous, collapse = "\n")
      tryCatch(py_compile_eval(pasted, capture = FALSE), error = handle_error)

      # now, handle the newest line of code submitted
      buffer$set(contents)
      code <- contents
      ready <- tryCatch(compiler(code), condition = identity)

      # a NULL return implies that we can accept more input
      if (is.null(ready))
        return()
    }

    # otherwise, we should have received a code output object
    # so we can just run the code submitted thus far
    buffer$clear()
    tryCatch(py_compile_eval(code, capture = FALSE), error = handle_error)

  }

  # notify the user we're entering the REPL (when requested)
  if (!quiet) {

    version <- paste(
      sys$version_info$major,
      sys$version_info$minor,
      sys$version_info$micro,
      sep = "."
    )

    # NOTE: we used to use sys.executable but that would report
    # the R process rather than the Python process
    config <- py_config()
    executable <- config$python

    fmt <- c(
      "Python %s (%s)",
      "Reticulate %s REPL -- A Python interpreter in R."
    )

    msg <- sprintf(
      paste(fmt, collapse = "\n"),
      version,
      executable,
      utils::packageVersion("reticulate")
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

# Check Whether the Python REPL is Active
#
# Check to see whether the Python REPL is active. This is primarily
# for use by R front-ends, which might want to toggle or affect
# the state of the Python REPL while it is running.
py_repl_active <- function() {
  .globals$py_repl_active
}
