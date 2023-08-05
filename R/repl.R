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
#' @param module An (optional) Python module to be imported before
#'   the REPL is launched.
#'
#' @param quiet Boolean; print a startup banner when launching the REPL? If
#'   `TRUE`, the banner will be suppressed.
#'
#' @param input Python code to be run within the REPL. Setting this can be
#'   useful if you'd like to drive the Python REPL programmatically.
#'
#' @seealso [py], for accessing objects created using the Python REPL.
#'
#' @section Magics: A handful of magics are supported in `repl_python()`:
#'
#' Lines prefixed with `!` are executed as system commands:
#'  - `!cmd --arg1 --arg2`: Execute arbitrary system commands
#'
#' Magics start with a `%` prefix. Supported magics include:
#'  - `%conda ...` executes a conda command in the active conda environment
#'  - `%pip ...` executes pip for the active python.
#'  - `%load`, `%loadpy`, `%run` executes a python file.
#'  - `%system`, `!!` executes a system command and capture output
#'  - `%env`: read current environment variables.
#'    - `%env name`: read environment variable 'name'.
#'    - `%env name=val`, `%env name val`: set environment variable 'name' to 'val'.
#'       `val` elements in `{}` are interpolated using f-strings (required Python >= 3.6).
#'  - `%cd <dir>` change working directory.
#'     - `%cd -`: change to previous working directory (as set by `%cd`).
#'     - `%cd -3`: change to 3rd most recent working directory (as set by `%cd`).
#'     - `%cd -foo/bar`: change to most recent working directory matching `"foo/bar"` regex
#'       (in history of directories set via `%cd`).
#'  - `%pwd`: print current working directory.
#'  - `%dhist`: print working directory history.
#'
#' Additionally, the output of system commands can be captured in a variable, e.g.:
#'  - `x = !ls`
#'
#'  where `x` will be a list of strings, consisting of
#'  stdout output split in `"\n"` (stderr is not captured).
#'
#'
#' @section Example:
#' ````
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
#' ````
#'
#' @importFrom utils packageVersion
#' @export
repl_python <- function(
  module  = NULL,
  quiet   = getOption("reticulate.repl.quiet", default = FALSE),
  input   = NULL)
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

  # split provided code on newlines
  if (!is.null(input))
    input <- unlist(strsplit(input, "\n", fixed = TRUE))

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
  buffer <- stack(mode = "character")

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
      error_message <- py_last_error()$message

      if (identical(.Platform$GUI, "RStudio") &&
          requireNamespace("cli", quietly = TRUE))
        error_message <- make_filepaths_clickable(error_message)

      message(error_message, appendLF = !endsWith(error_message, "\n"))
    }
    failed
  }

  handle_interrupt <- function(condition) {
    # swallow interrupts -- don't allow interrupted Python code to
    # exit the REPL; we should only exit when an interrupt is sent
    # when no Python code is executing
  }

  repl <- function() {

    # flush stdout, stderr on each REPL iteration
    on.exit(py_flush_output(), add = TRUE)

    # read input (either from user or from code)
    prompt <- if (buffer$empty()) ">>> " else "... "
    if (is.null(input)) {
      contents <- readline(prompt = prompt)
    } else if (length(input)) {
      contents <- input[[1L]]
      input <<- tail(input, n = -1L)
      writeLines(paste(prompt, contents), con = stdout())
    } else {
      quit_requested <<- TRUE
      return()
    }

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

    # run hook provided by front-end, to notify that we're now busy
    hook <- getOption("reticulate.repl.busy")
    if (is.function(hook)) {

      # run once now to indicate we're about to run
      status <- tryCatch(hook(TRUE), error = identity)
      if (inherits(status, "error"))
        warning(status)

      # run again on exit to indicate we're done
      on.exit({
        status <- tryCatch(hook(FALSE), error = identity)
        if (inherits(status, "error"))
          warning(status)
      }, add = TRUE)

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
      if (grepl("(^[\\#].*)[?]\\s*$", trimmed, perl = TRUE)) {
        replaced <- sub("[?]\\s*$", "", trimmed)
        code <- sprintf("help(\"%s\")", replaced)
        py_run_string(code)
        return()
      }

      if (getOption("reticulate.repl.use_magics", TRUE)) {

        # expand any "!!" as system commands that capture output
        trimmed <- gsub("!!", "%system ", trimmed)

        # user intends to capture output from system command in var
        # e.g.:   x = !ls
        if (grepl("^[[:alnum:]_.]\\s*=\\s*!", trimmed))
          trimmed <- sub("=\\s*!", "= %system ", trimmed)

        # magic
        if (grepl("^%", trimmed)) {
          py$`_` <- .globals$py_last_value <- invoke_magic(trimmed)
          return()
        }

        # system
        if (grepl("^!", trimmed)) {
          system(str_drop_prefix(trimmed, "!"))
          return()
        }

        # capture output from magic command in var
        #   # e.g.:   x = %env USER
        if (grepl("^[[:alnum:]_.]\\s*=\\s*%", trimmed)) {
          s <- str_split1_on_first(trimmed, "\\s*=\\s*")
          target <- s[[1]]
          magic <- str_drop_prefix(s[2L], "%")
          py$`_` <- .globals$py_last_value <- invoke_magic(magic)
          py_run_string(sprintf("%s = _", target), local = FALSE, convert = FALSE)
          return()
        }
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

      tryCatch(
        py_compile_eval(pasted, capture = FALSE),
        error = handle_error,
        interrupt = handle_interrupt
      )

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

    tryCatch(
      py_compile_eval(code, capture = FALSE),
      error = handle_error,
      interrupt = handle_interrupt
    )

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
      "Reticulate %s REPL -- A Python interpreter in R.",
      "Enter 'exit' or 'quit' to exit the REPL and return to R."
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

    tryCatch(repl(), interrupt = identity)

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


invoke_magic <- function(command) {
  stopifnot(is.character(command), length(command) == 1)
  command <- str_drop_prefix(command, "%")

  m <- str_split1_on_first(command, "\\s+")
  cmd <- m[1]
  args <- m[-1]


  if (cmd == "pwd") {
    if (length(args))
      stop("%pwd magic takes no arguments, received: ", command)
    dir <- getwd()
    cat(dir, "\n")
    return(invisible(dir))
  }

  # in IPython, this is stored in __main__._dh as a python list
  # we avoid polluting `__main__` and also lazily create the history,
  # also, this can only track changes made from `repl_python()`.
  get_dhist <- function() {
    dh <- .globals$magics_state$wd_history
    if (is.null(dh)) {
      .globals$magics_state <- new.env(parent = emptyenv())
      dh <- import("collections")$deque(list(getwd()), 200L)
      .globals$magics_state$wd_history <- dh
    }
    dh
  }

  if (cmd == "cd") {
     hist <- get_dhist()

    if (length(args) != 1)
       stop("%cd magic takes 1 argument, received: ", command)

     dir <- gsub("[\"']", "", args)
     # strings auto complete as fs locations in RStudio IDE, so as a convenience
     # we accept quoted file paths and unquote them here.

     setwd2 <- function(dir) {
       old_wd <- setwd(dir)
       new_wd <- getwd()
       cat(new_wd, "\n", sep = "")
       hist$append(new_wd)
       invisible(old_wd)
     }

    if (startsWith(args, "-")) {
      if (args == "-") {
        dir <- hist[-2L]
      } else if (grepl("-[0-9]+$", args)) {
        dir <- hist[as.integer(args)]
      } else {
        # partial matching by regex
        hist <- import_builtins()$list(hist)
        re <- str_drop_prefix(args, "-")
        if (is_windows())
          re <- gsub("[/]", "\\", re, fixed = TRUE)
        dir <- grep(re, hist, perl = TRUE, value = TRUE)
        if (!length(dir))
          stop("No matching directory found in history for ", dQuote(re), ".",
               "\nSee history with %dhist")

        dir <- dir[[length(dir)]] # pick most recent match
      }
      # not implemented, -b bookmarks, -q quiet
    } else
      dir <- args

    return(setwd2(dir))
  }

  if (cmd == "dhist") {
    hist <- get_dhist()
    hist <- import_builtins()$list(hist)
    cat("Directory history:\n- ")
    cat(hist, sep = "\n- ")
    cat("\n")
    return(invisible(hist))
  }

  if (cmd == "conda") {
    info <- get_python_conda_info(py_exe())
    return(conda_run2(cmd_line = paste("conda", args),
                      conda = info$conda,
                      envname = info$root))
  }

  if (cmd == "pip") {
    if (is_conda_python(py_exe())) {
      info <- get_python_conda_info(py_exe())
      return(conda_run2(cmd_line = paste("pip", args),
                        conda = info$conda,
                        envname = info$root))
    } else {
      args <- shQuote(strsplit(args, "\\s+")[[1]])
      system2(py_exe(), c("-m", "pip", args))
    }
    return()
  }

  if (cmd == "env") {

    if (!length(args))
      return(print(Sys.getenv()))

    if (grepl("=|\\s", args)) # user setting var
      args <- str_split1_on_first(args, "=|\\s+")
    else {
      print(val <- Sys.getenv(args))
      return(val)
    }

    new_val <- args[[2]]
    if (grepl("\\{.*\\}", new_val) && py_version() >= "3.6") {
      #interpolate as f-strings
      new_val <- py_eval(sprintf('f"%s"', new_val))
    }
    names(new_val) <- args[[1]]
    do.call(Sys.setenv, as.list(new_val))
    cat(sprintf("env: %s=%s\n", names(new_val), new_val))
    return(invisible(new_val))
    # not implemented: bash-style $var expansion
  }

  if (cmd %in% c("load", "loadpy", "run")) {
    # only supports sourcing a python file in __main__
    # not implemented:
    # -r line ranges, -s specific symbols,
    # reexecution of symbols from history,
    # reexecution of namespace objects annotated by ipython shell with original source
    # ipython extensions
    file <- gsub("[\"']", "", args)
    if (!file.exists(file))
      stop("Python file not found: ", file)
    py_run_file(file, local = FALSE, convert = FALSE)
    return()
  }

  if (cmd %in% c("system", "sx")) {
    if (is_windows())
      return(shell(args, intern = TRUE))
    else
      return(as.list(system(args, intern = TRUE)))
  }

  stop("Magic not implemented: ", command)
}



#' IPython console
#'
#' Launch IPython console app.
#'
#' See https://ipython.readthedocs.io/ for features.
#'
#' @keywords internal
ipython <- function() {
  ensure_python_initialized("IPython")

  # set flag for frontend
  .globals$py_repl_active <- TRUE
  on.exit({
    .globals$py_repl_active <- FALSE
  }, add = TRUE)

  # don't pollute R history w/ python commands
  # (IPython keeps track of it's own history)
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
  }

  # Not implemented,
  #  - custom startup banner
  # RStudio IDE support for:
  #  - image display
  #  - composition of multi-line code blocks

  import("rpytools.ipython")$start_ipython()
}
