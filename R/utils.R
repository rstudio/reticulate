
`%||%` <- function(x, y) if (is.null(x)) y else x

get_r_trace <- function(maybe_use_cached = FALSE, trim_tail = 1L) {
  # this function, `get_r_trace()`, can get called repeatedly as a stack is
  # being unwound, at each transition between r -> py frames if python 'lost'
  # the r_trace attr as part of handling the exception. Here we make sure we
  # don't return a truncated version of the same trace if this function is being
  # called as a stack is unwinding due to an exception propogating.

  # note, Exceptions typically have r_trace and r_call attrs set on creation,
  # but those can be lost or discarded by python code. There are two
  # common scenarios where the r_trace attr is missing after raising the
  # exception in python and then recatching it in R as the stack is unwound:

  # 1: statements in python like `raise from` will nestle the exception
  # containing the r_trace into chain of `__context__` attributes. This is
  # scenario handled in C++ py_fetch_error() by walking the chain of
  # `__context__` attributes and copying the original r_trace over to the head
  # of the exception chain.

  # 2: tensorflow.autograph, **completely discards the original exception,
  # clears the error, then raises a new exception** when transforming a
  # function. The new exception raised is a de-novo constructed exception that
  # containing a half-hearted text summary + pre-formatted python traceback of
  # the original exception. This means that if we create an exception object
  # here with an r_trace attr, and then `raise` it in python from the wrapper
  # created by rpytools.call, when we next encounter a propogating exception at
  # the next r<->py boundry as the stack is being unwound, it is a
  # *different* exception (new memory address, potentially different type(),
  # none of the original attributes, with no way to recover the original
  # attributes we want, like r_trace). This means that attaching the r_trace to
  # the python exception and passing it through the python runtime cannot be
  # relied on. (at least not with with tf.autograph, or things that use it like
  # keras. Probably other approaches that involve rewriting or modifying python
  # ast like numba and friends will fail similarly).

  # Hence, this approach, where, to mitigate the scenario where arbitrary python
  # code lost the r_trace, we cache the r_trace in R, and then try to be smart
  # about pairing the R trace with the correct python exception when presenting
  # the error to the user. Note, pairing an r trace with the correct exception
  # is tricky and bound to fail in edge cases too, but w.r.t. tradeoffs, the
  # failure mode will be more forgiving; the user will be presented with an
  # r_trace that is too long rather than too short.

  # (rlang traces are dataframes.)
  t <- rlang::trace_back() # bottom=2 to omit this `save_r_trace()` frame
  t <- t[1L:(nrow(t) - trim_tail), ] # https://github.com/r-lib/rlang/issues/1620

  ## the rlang trace contains calls mangled for pretty printing. Unfortunately,
  ## the mangling is too aggressive, the actual call is frequently needed to
  ## track down where an error occurred.
  t$full_call <- sys.calls()[seq_len(nrow(t))]

  # Drop reticulate internal frames that are not useful to the user
  ## (this works, except [ method for traces does not adjust the parent
  ## correctly when slicing out frames where parent == 0, and
  ## then the tree that gets printed is not useful.
  ## TODO: file an issue with rlang)
  # i <- 1L
  # while(i < nrow(t)) {
  #   if(identical(t$call[[i]][[1L]], quote(call_r_function))) {
  #     # drop frames:
  #     # withRestarts(withCallingHandlers(return(list(do.call(fn, c(args, named_args)), NULL)), python.builtin.BaseException = function(e) {     r_tra…
  #     # withOneRestart(expr, restarts[[1L]])
  #     # doWithOneRestart(return(expr), restart)
  #     # withCallingHandlers(return(list(do.call(fn, c(args, named_args)), NULL)), python.builtin.BaseException = function(e) {     r_trace <- py_get_…
  #     # do.call(fn, c(args, named_args))
  #     i <- i + 1L
  #     t <- t[-seq.int(from = i, length.out = 5L), ]
  #   }
  #
  #   # drop py_call_impl() frame
  #   else if(identical(t$call[[i]][[1L]], quote(py_call_impl))) {
  #     t <- t[-i, ]
  #   }
  #
  #   else {
  #     i <- i + 1L
  #   }
  # }

  if(!maybe_use_cached)
    return((.globals$last_r_trace <- t))

  ot <- .globals$last_r_trace

  if (# no previously cached trace
      is.null(ot) ||

      # new trace is longer than previously cached trace, must be new
      nrow(t) >= nrow(ot) ||

      # new trace is not a subset of previously cached trace
      !identical(t, ot[seq_len(nrow(t)), ])) {
    .globals$last_r_trace <- t
  }

  invisible(.globals$last_r_trace)
}


call_r_function <- function(fn, args, named_args) {
  withRestarts(

    withCallingHandlers(

      return(list(do.call(fn, c(args, named_args)), NULL)),

      python.builtin.BaseException = function(e) {
        # check if rethrowing an exception that we've already seen
        # and if so, make sure the r_trace attr is still present
        r_trace <- as_r_value(py_get_attr(e, "trace", TRUE))
        if(is.null(r_trace)) {
          r_trace <- get_r_trace(maybe_use_cached = TRUE, trim_tail = 2)
          py_set_attr(e, "trace", py_capsule(r_trace))
        }

        if(!py_has_attr(e, "call")) {
          py_set_attr(e, "call", py_capsule(r_trace$full_call[[nrow(r_trace)]]))
        }

        invokeRestart("raise_py_exception", e)
      },

      interrupt = function(e) {
        invokeRestart("raise_py_exception", "KeyboardInterrupt")
      },

      error = function(e) {
        # we're encountering an R error that has not yet been converted to Python
        .globals$last_r_trace <- e$trace <-
          e$trace %||% get_r_trace(maybe_use_cached = FALSE, trim_tail = 2)
        invokeRestart("raise_py_exception", e)
      }
    ), # end withCallingHandlers()

    raise_py_exception = function(e) {
      list(NULL, e)
    }
  ) # end withRestarts()
}


as_r_value <- function(x) py_to_r(x)

#' @export
r_to_py.error <- function(x, convert = FALSE) {
  if(inherits(x, "python.builtin.object")) {
    assign("convert", convert, envir = as.environment(x))
    return(x)
  }

  bt <- import_builtins(convert = convert)
  e <- bt$RuntimeError(conditionMessage(x))

  for (nm in names(x))
    py_set_attr(e, nm, py_capsule(x[[nm]]))

  py_set_attr(e, "r_class", py_capsule(class(x)))

  e
}


#' @export
`$.python.builtin.BaseException` <- function(x, name) {
  if(identical(name, "call")) {
    out <- if(typeof(x) == "list") unclass(x)[["call"]]
    return(out %||% as_r_value(py_get_attr(x, "call", TRUE)))
  }

  if(identical(name, "message")) {
    out <- if(typeof(x) == "list") unclass(x)[["message"]]
    return(out %||% conditionMessage_from_py_exception(x))
  }

  py_maybe_convert(py_get_attr(x, name, TRUE), py_has_convert(x))
}

#' @export
`[[.python.builtin.BaseException` <- `$.python.builtin.BaseException`


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

defer <- function(expr, envir = parent.frame(), priority = c("first", "last")) {
  thunk <- as.call(list(function() expr))
  after <- priority == "last"
  do.call(base::on.exit, list(thunk, TRUE, after), envir = envir)
}

#' @importFrom utils head
disable_conversion_scope <- function(object) {
  # Though this is not part of the exported API, there are external packages
  # that reach in with ::: to use this function. (e.g., {zellkonverter} on
  # Bioconductor). Take care that symbols like `py_set_convert` and `object`
  # don't need to be in the on.exit() expression search path.

  if (!is_py_object(object) || !py_get_convert(object))
    return(FALSE)

  envir <- parent.frame()
  cl <- as.call(c(py_set_convert, object, TRUE))

  py_set_convert(object, FALSE)
  do.call(on.exit, list(cl, add = TRUE), envir = envir)

  TRUE
}

local_conversion_scope <- function(object, value, envir = parent.frame()) {
  if(py_get_convert(object) == value)
    return()

  py_set_convert(object, value)
  cl <- call("py_set_convert", object, !value)
  do.call(on.exit, list(cl, add = TRUE), envir = envir)
}



py_compile_eval <- function(code, compile_mode = "single", capture = TRUE) {

  builtins <- import_builtins(convert = TRUE)
  sys <- import("sys", convert = TRUE)

  # allow 'globals' and 'locals' to both point at main module, so that
  # evaluated code updates references there as well
  main <- import_main(convert = FALSE)
  globals <- locals <- py_get_attr(main, "__dict__")


  # Python's command compiler complains if the only thing you submit
  # is a comment, so detect that case first
  is_comments_only <- local({
    code <- trimws(strsplit(code, "\n", fixed = TRUE)[[1]])
    code <- code[nzchar(code)]
    all(startsWith(code, "#"))
  })
  if (is_comments_only)
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
  ex <- .globals$py_last_exception
  rtb <- .globals$last_r_trace
  tryCatch(
    py_eval("_", convert = FALSE),
    error = function(e) {
      .globals$py_last_exception <- ex
      .globals$last_r_trace <- rtb
      py_none()
    }
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
  lhsi <- suppressWarnings(c(file.info(lhs, extra_cols = FALSE)))
  rhsi <- suppressWarnings(c(file.info(rhs, extra_cols = FALSE)))
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
    # on windows, normalizePath("") returns "C:/"
    if(isFALSE(nzchar(path))) return("")
    normalizePath(path, winslash = "/", mustWork = FALSE)
  } else {
    # on linux/mac, we protect against `normalizePath()` resolving
    # python binaries that are symbolic links, as encountered in python venvs.
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

is_r_cmd_check <- function() {

  # if NOT_CRAN is set, this is likely devtools::check() -- allow it
  not_cran <- Sys.getenv("NOT_CRAN", unset = NA)
  if (identical(not_cran, "true"))
    return(FALSE)

  # if _R_CHECK_PACKAGE_NAME_ is set, then we must be running R CMD check
  package_name <- Sys.getenv("_R_CHECK_PACKAGE_NAME_", unset = NA)
  if (!is.na(package_name))
    return(TRUE)

  # does not appear to be R CMD check
  FALSE

}

stack <- function(mode = "list") {

  .data <- vector(mode)

  object <- list(

    set = function(data) {
      .data <<- data
    },

    push = function(...) {
      dots <- list(...)
      for (data in dots) {
        if (is.null(data))
          .data[length(.data) + 1] <<- list(NULL)
        else
          .data[[length(.data) + 1]] <<- data
      }
    },

    pop = function() {
      item <- .data[[length(.data)]]
      length(.data) <<- length(.data) - 1
      item
    },

    peek = function() {
      .data[[length(.data)]]
    },

    contains = function(data) {
      data %in% .data
    },

    empty = function() {
      length(.data) == 0
    },

    clear = function() {
      .data <<- list()
    },

    data = function() {
      .data
    }

  )

  object

}

get_hooks_list <- function(name) {
  hooks <- getHook(name)
  if (!is.list(hooks))
    hooks <- list(hooks)
  hooks
}

deparse1 <- function(expr, width.cutoff = 500L) {
  paste(deparse(expr, width.cutoff), collapse = " ")
}

isTRUE <- function(x) {
  is.logical(x) && length(x) == 1L && !is.na(x) && x
}

isFALSE <- function(x) {
  is.logical(x) && length(x) == 1L && !is.na(x) && !x
}

is_string <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x)
}

home <- function() {
  path.expand("~")
}

aliased_path <- function(path) {

  home <- home()
  if (!nzchar(home))
    return(path)

  home <- chartr("\\", "/", home)
  path <- chartr("\\", "/", path)

  match <- regexpr(home, path, fixed = TRUE, useBytes = TRUE)
  path[match == 1] <- file.path("~", substring(path[match == 1], nchar(home) + 2L))

  path

}

pretty_path <- function(path) {
  encodeString(aliased_path(path), quote = '"')
}

heredoc <- function(text) {

  # remove leading, trailing whitespace
  trimmed <- gsub("^\\s*\\n|\\n\\s*$", "", text)

  # split into lines
  lines <- strsplit(trimmed, "\n", fixed = TRUE)[[1L]]

  # compute common indent
  indent <- regexpr("[^[:space:]]", lines)
  common <- min(setdiff(indent, -1L))

  # remove common indent
  paste(substring(lines, common), collapse = "\n")

}

dir.exists <- function(paths) {
  utils::file_test("-d", paths)
}

str_split1_on_first <- function(x, pattern, ...) {
  stopifnot(length(x) == 1, is.character(x))
  regmatches(x, regexpr(pattern, x, ...), invert = TRUE)[[1L]]
}

str_drop_prefix <- function(x, prefix) {

  if (is.character(prefix)) {
    if (!startsWith(x, prefix))
      return(x)
    prefix <- nchar(prefix)
  }

  substr(x, as.integer(prefix) + 1L, nchar(x))

}

`append<-` <- function(x, value) {
  c(x, value)
}

if (getRversion() < "3.3.0") {

startsWith <- function(x, prefix) {
  if (!is.character(x) || !is.character(prefix))
    stop("non-character object(s)")
  suppressWarnings(substr(x, 1L, nchar(prefix)) == prefix)
}

endsWith <- function(x, suffix) { # needed for R < 3.3
  if (!is.character(x) || !is.character(suffix))
    stop("non-character object(s)")
  n <- nchar(x)
  suppressWarnings(substr(x, n - nchar(suffix) + 1L, n) == suffix)
}

trimws <- function (x, which = c("both", "left", "right"),
                    whitespace = "[ \t\r\n]") {
  which <- match.arg(which)
  mysub <- function(re, x)
    sub(re, "", x, perl = TRUE)
  switch( which,
    left = mysub(paste0("^", whitespace, "+"), x),
    right = mysub(paste0(whitespace, "+$"), x),
    both = mysub(paste0(whitespace, "+$"),
                 mysub(paste0("^", whitespace, "+"), x))
  )
}


}


debuglog <- function(fmt, ...) {
  msg <- sprintf(fmt, ...)
  cat(msg, file = "/tmp/reticulate.log", sep = "\n", append = TRUE)
}

system2t <- function(command, args, ...) {
  # system2, with a trace
  # mimic bash's set -x usage of a "+" prefix for now
  # maybe someday take a dep on {cli} and make it prettier
  message(paste("+", maybe_shQuote(command), paste0(args, collapse = " ")))
  system2(command, args, ...)
}

maybe_shQuote <- function(x) {
  needs_quote <- !grepl("^[[:alnum:]/._-]+$", x)
  if(any(needs_quote))
    x[needs_quote] <- shQuote(x[needs_quote])
  x
}


rm_all_reticulate_state <- function(external = FALSE) {

  rm_rf <- function(...)
    unlink(path.expand(c(...)), recursive = TRUE, force = TRUE)

  if (external) {
    if (!is.null(uv <- uv_binary(FALSE))) {
      system2(uv, c("cache", "clean"))
      rm_rf(system2(uv, c("python", "dir"),
                    env = "NO_COLOR=1", stdout = TRUE))
      # rm_rf(system2(uv, c("tool", "dir"),
      #               env = "NO_COLOR=1", stdout = TRUE))
    }

    if (nzchar(Sys.which("pip3")))
      system2("pip3", c("cache", "purge"))
  }

  rm_rf(user_data_dir("r-reticulate", NULL))
  rm_rf(user_data_dir("r-miniconda", NULL))
  rm_rf(user_data_dir("r-miniconda-arm64", NULL))
  rm_rf(rappdirs::user_cache_dir("r-reticulate", NULL))
  rm_rf(miniconda_path_default())
  rm_rf(virtualenv_path("r-reticulate"))
  for (venv in virtualenv_list()) {
    if (startsWith(venv, "r-"))
      virtualenv_remove(venv, confirm = FALSE)
  }
}


user_data_dir <- function(...) {
  expand_env_vars(rappdirs::user_data_dir(...))
}

expand_env_vars <- function(x) {
  # We need to expand some env vars here, until
  # Rstudio server is patched.
  # https://github.com/rstudio/rstudio-pro/issues/2968
  # The core issue is RStudio Server shell expands some env vars, but
  # doesn't propogate those expanded env vars to the user R sessions
  # e.g., https://docs.posit.co/ide/server-pro/1.4.1722-1/server-management.html#setting-environment-variables
  # suggests adminst set XDG_DATA_HOME=/mnt/storage/$USER
  # that is correctly expanded by rstudio server here:
  # https://github.com/rstudio/rstudio/blob/55c42e8d9c0df19a6566000f550a0fa6dc519899/src/cpp/core/system/Xdg.cpp#L160-L178
  # but then not propogated to the user R session.
  # https://github.com/rstudio/reticulate/issues/1513

  if(!grepl("$", x, fixed = TRUE))
    return(x)
  delayedAssign("info", Sys.info())
  delayedAssign("HOME", Sys.getenv("HOME") %""% path.expand("~"))
  delayedAssign("USER", Sys.getenv("USER") %""% info[["user"]])
  delayedAssign("HOSTNAME", Sys.getenv("HOSTNAME") %""% info[["nodename"]])
  for (name in c("HOME", "USER", "HOSTNAME")) {
    if (grepl(name, x, fixed = TRUE)) {
      x <- gsub(sprintf("$%s", name), get(name), x, fixed = TRUE)
      x <- gsub(sprintf("${%s}", name), get(name), x, fixed = TRUE)
    }
  }
  x
}

`%""%` <- function(x, y) if(identical(x, "")) y else x
