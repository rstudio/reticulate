
`%||%` <- function(x, y) if (is.null(x)) y else x

traceback_enabled <- function() {

  # if there is specific option set then respect it
  reticulate_traceback_option <- getOption("reticulate.traceback", default = NULL)
  if (!is.null(reticulate_traceback_option))
    return(isTRUE(reticulate_traceback_option))

  # determine whether rstudio python traceback support is available
  # and whether rstudio tracebacks are currently enabled
  rstudio_has_python_tracebacks <- exists(".rs.getActivePythonStackTrace",
                                          mode = "function")
  if (rstudio_has_python_tracebacks) {

    error_option_code <- deparse(getOption("error"))
    error_option_code_has <- function(pattern) {
      any(grepl(pattern, error_option_code))
    }
    rstudio_traceback_enabled <- error_option_code_has("\\.rs\\.recordTraceback")

    # if it is then we disable tracebacks
    if (rstudio_traceback_enabled)
      return(FALSE)
  }

  # default to tracebacks enabled
  TRUE
}

clear_error_handler <- function(retvalue = NA) {
  function(e) {
    py_clear_last_error()
    if (!is.null(retvalue) && is.na(retvalue))
      e
    else
      retvalue
  }
}

as_r_value <- function(x) {
  if (inherits(x, "python.builtin.object"))
    py_to_r(x)
  else
    x
}

yoink <- function(package, symbol) {
  do.call(":::", list(package, symbol))
}

defer <- function(expr, envir = parent.frame()) {
  call <- substitute(
    evalq(expr, envir = envir),
    list(expr = substitute(expr), envir = parent.frame())
  )
  do.call(base::on.exit, list(substitute(call), add = TRUE), envir = envir)
}

#' @importFrom utils head
disable_conversion_scope <- function(object) {

  if (!inherits(object, "python.builtin.object"))
    return(FALSE)

  envir <- as.environment(object)
  if (exists("convert", envir = envir, inherits = FALSE)) {
    convert <- get("convert", envir = envir)
    assign("convert", FALSE, envir = envir)
    defer(assign("convert", convert, envir = envir), envir = parent.frame())
  }

  TRUE
}

new_stack <- function() {

  (function() {

    .data <- list()

    methods <- list(
      clear  = function() { .data <<- character() },
      data   = function() { .data },
      empty  = function() { length(.data) == 0 },
      length = function() { length(.data) },
      push   = function(line) { .data[[length(.data) + 1]] <<- line },
      peek   = function() { .data[[length(.data)]] },
      pop    = function() { .data <<- utils::head(.data, n = -1) },
      set    = function(data) { .data <<- data }
    )

    list2env(methods)

  })()

}

py_compile_eval <- function(code, compile_mode = "single", capture = TRUE) {

  builtins <- import_builtins(convert = TRUE)
  sys <- import("sys", convert = TRUE)

  # allow 'globals' and 'locals' to both point at main module, so that
  # evaluated code updates references there as well
  globals <- py_eval("globals()", convert = FALSE)
  locals <- globals

  # Python's command compiler complains if the only thing you submit
  # is a comment, so detect that case first
  if (grepl("^\\s*#", code))
    return(TRUE)

  # Python is picky about trailing whitespace, so ensure only a single
  # newline follows the code to be submitted
  code <- sub("\\s*$", "\n", code)

  # compile and eval the code -- use 'single' to auto-print statements
  # as they are evaluated, or 'exec' to avoid auto-print
  compiled <- builtins$compile(code, '<string>', compile_mode)
  if (capture) {
    output <- py_capture_output(builtins$eval(compiled, globals, locals))
  } else {
    builtins$eval(compiled, globals, locals)
    output <- NULL
  }

  # save the value that was produced
  .globals$py_last_value <- py_last_value()

  # py_capture_output can append an extra trailing newline, so remove it
  if (!is.null(output) && grepl("\n{2,}$", output))
    output <- sub("\n$", "", output)

  # and return
  invisible(output)
}

py_last_value <- function() {
  tryCatch(
    py_eval("_", convert = FALSE),
    error = function(e) r_to_py(NULL)
  )
}

python_binary_path <- function(dir) {

  # check for condaenv
  if (is_condaenv(dir)) {
    suffix <- if (is_windows()) "python.exe" else "bin/python"
    return(file.path(dir, suffix))
  }

  # check for virtualenv
  if (is_virtualenv(dir)) {
    suffix <- if (is_windows()) "Scripts/python.exe" else "bin/python"
    return(file.path(dir, suffix))
  }

  # check for directory containing Python
  suffix <- if (is_windows()) "python.exe" else "python"
  if (file.exists(file.path(dir, suffix)))
    return(file.path(dir, suffix))

  stop("failed to discover Python binary associated with path '", dir, "'")

}

# prepends entries to the PATH (either moving or adding them as appropriate)
# and returns the previously-set PATH
path_prepend <- function(entries) {
  oldpath <- Sys.getenv("PATH")
  if (length(entries)) {
    entries <- path.expand(entries)
    splat <- strsplit(oldpath, split = .Platform$path.sep, fixed = TRUE)[[1]]
    newpath <- c(entries, setdiff(splat, entries))
    Sys.setenv(PATH = paste(newpath, collapse = .Platform$path.sep))
  }
  oldpath
}

# note: normally, we'd like to compare paths with normalizePath() but
# that does not normalize for case on Windows by default so we fall back
# to a heuristic (note: false positives are possible but we can accept
# those in the contexts where this function is used)
file_same <- function(lhs, rhs) {

  # check if paths are identical as-is
  if (identical(lhs, rhs))
    return(TRUE)

  # check if paths are identical after normalization
  lhs <- normalizePath(lhs, winslash = "/", mustWork = FALSE)
  rhs <- normalizePath(rhs, winslash = "/", mustWork = FALSE)
  if (identical(lhs, rhs))
    return(TRUE)

  # check if file info is the same
  lhsi <- c(file.info(lhs, extra_cols = FALSE))
  rhsi <- c(file.info(rhs, extra_cols = FALSE))
  fields <- c("size", "isdir", "mode", "mtime", "ctime")
  if (identical(lhsi[fields], rhsi[fields]))
    return(TRUE)

  # checks failed; return FALSE
  FALSE

}

# normalize a path without following symlinks
canonical_path <- function(path) {

  # on windows we normalize the whole path to avoid
  # short path components leaking in
  if (is_windows()) {
    normalizePath(path, winslash = "/", mustWork = FALSE)
  } else {
    file.path(
      normalizePath(dirname(path), winslash = "/", mustWork = FALSE),
      basename(path)
    )
  }

}

enumerate <- function(x, f, ...) {
  n <- names(x)
  lapply(seq_along(x), function(i) {
    f(n[[i]], x[[i]], ...)
  })
}

is_interactive <- function() {
  
  # detect case where RStudio is being used, but reticulate is being
  # executed as part of a user's .Rprofile (for example). in this case,
  # we aren't really interactive as we are unable to respond to readline
  # requests, and so we want to avoid miniconda prompts
  rstudio <- !is.na(Sys.getenv("RSTUDIO", unset = NA))
  gui <- .Platform$GUI
  if (rstudio && !identical(gui, "RStudio"))
    return(FALSE)
  
  # otherwise, use base implementation
  interactive()
  
}
