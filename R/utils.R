
`%||%` <- function(x, y) if (is.null(x)) y else x

get_r_trace <- function(maybe_use_cached = FALSE, trim_tail = 1) {
  # this function, `get_r_trace()`, can get called repeatedly as a stack is being
  # unwound, at each transition between r -> py frames if python 'lost' the r_trace attr
  # as part of handling the exceptoin. Here we make sure we
  # don't return a truncated version of the same trace if this function is being called
  # as a stack is unwinding due to an exception propogating.

  # note, an earlier approach to capturing the R trace  saved a py_capsule() of
  # the r_trace as an attribute of the python exception, allowing it to traverse
  # the r<->py boundary during stack unwinding. This approach ran into two
  # issues:

  # 1: statements in python like `raise from` would nestle the actual r_trace
  # under a chain of `__context__` attributes. This was cumbersome, but easily
  # solvable by walking the chain of `__context__` attributes and copying the
  # original r_trace over to the head of the exception chain.

  # 2: tensorflow.autograph, **completely discards the original exception** when
  # transforming a function, replacing it with a de-novo constructed exception
  # that contains a half-hearted text summary + pre-formatted python traceback
  # contained in the new exception object it passes. This means that if we
  # create an exception object here and then `raise` it in python from the
  # wrapper created by rpytools.call, when we encounter a propogating exception at the
  # next r<->py boundry as the stack is being unwound, it is a *different*
  # exception (new memory address, potentially different type(), none of the
  # original attributes, with no way to recover the original attributes we want,
  # like r_trace). This means that attaching the r_trace to the python exception
  # and passing it through the python runtime is not going to work (at least
  # with with tf.autograph, or things that use it like keras. Probably other
  # approaches that involve rewriting or modifying python ast like numba and
  # friends will fail similarly).

  # Hence, this approach, where, to avoid giving arbitrary python code an
  # opportunity to lose the r_trace, we cache the r_trace in R, and then try to
  # be smart about pairing the R trace with the correct python exception when
  # presenting the error to the user. Note, pairing an r trace with the correct
  # exception is tricky and bound to fail in edge cases too, but w.r.t.
  # tradeoffs, the failure mode will be more forgiving; the user will be
  # presented with an r_trace that is too long rather than too short.

  # (rlang traces are dataframes.)
  t <- rlang::trace_back() # bottom=2 to omit this `save_r_trace()` frame
  t <- t[1:(nrow(t) - trim_tail), ] # https://github.com/r-lib/rlang/issues/1620

  ## the rlang trace contains calls mangled for pretty printing.
  ## Unfortunately, the mangling is too aggressive, the actual call is frequently needed
  ## to track down where an error occurred.
  t$full_call <- sys.calls()[seq_len(nrow(t))]

  # Drop reticulate internal frames that are not useful to the user
  ## (this works, except [ method for traces does not adjust the parent
  ## correctly when slicing out frames where parent == 0, and
  ## then the tree that gets printed is not useful.
  ## TODO: file an issue with rlang
  # i <- 1L
  # while(i < nrow(t)) {
  #     # drop frames:
  #     # withRestarts(withCallingHandlers(return(list(do.call(fn, c(args, named_args)), NULL)), python.builtin.BaseException = function(e) {     r_tra…
  #     # withOneRestart(expr, restarts[[1L]])
  #     # doWithOneRestart(return(expr), restart)
  #     # withCallingHandlers(return(list(do.call(fn, c(args, named_args)), NULL)), python.builtin.BaseException = function(e) {     r_trace <- py_get_…
  #     # do.call(fn, c(args, named_args))
  #   if(identical(t$call[[i]][[1L]], quote(call_r_function))) {
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

printtrace <- function(x) {
  tibble::as_tibble(x) |>
    dplyr::mutate(call2 = sapply(call, deparse1)) |>
    print(n = Inf)
}


call_r_function <- function(fn, args, named_args) {
  withRestarts(

    withCallingHandlers(

      return(list(do.call(fn, c(args, named_args)), NULL)),

      python.builtin.BaseException = function(e) {
        # we're throwing a python exception
        # check if we're rethrowing an exception that we've already seen
        # and if so, make sure the r_trace attr is still present
        r_trace <- as_r_value(py_get_attr(e, "r_trace", TRUE))
        if(is.null(r_trace)) {
          r_trace <- get_r_trace(maybe_use_cached = TRUE, trim_tail = 2)
          py_set_attr(e, "r_trace", py_capsule(r_trace))
        }

        if(!py_has_attr(e, "r_call")) {
          if(is.null(r_trace))
            r_trace <- get_r_trace(maybe_use_cached = TRUE, trim_tail = 2)
          py_set_attr(e, "r_call", py_capsule(r_trace$full_call[[nrow(r_trace)]]))
        }

        invokeRestart("raise_py_exception", e)
      },

      interrupt = function(e) {
        invokeRestart("raise_py_exception", "KeyboardInterrupt")
      },

      error = function(e) {
        # we're encountering an R error that has not yet been converted to Python
        trace <- e$trace
        if(is.null(trace))
          trace <- get_r_trace(maybe_use_cached = FALSE, trim_tail = 2)
        e$trace <- .globals$last_r_trace <- trace
        invokeRestart("raise_py_exception", e)
      }
    ), # withCallingHandlers()

    raise_py_exception = function(e) {
      list(NULL, e)
    }
  ) # withRestarts()
}


as_r_value <- function(x) if(inherits(x, "python.builtin.object")) py_to_r(x) else x


#' @export
r_to_py.error <- function(x, convert = FALSE) {
  if(inherits(x, "python.builtin.object")) {
    assign("convert", convert, envir =  as.environment(x))
    return(x)
  }

  e <- import_builtins(convert = convert)$RuntimeError(conditionMessage(x))

  for (nm in setdiff(names(x), c("call", "message")))
    py_set_attr(e, paste0("r_", nm), py_capsule(x[[nm]]))

  py_set_attr(e, "r_call", conditionCall(x))
  py_set_attr(e, "r_class", class(x))

  e
}

#' @export
conditionCall.python.builtin.BaseException <- function(c) {
  as_r_value(py_get_attr(c, "r_call", TRUE))
}

#' @export
conditionMessage.python.builtin.BaseException <- function(c) {
  conditionMessage_from_py_exception(c)
}

#' @export
print.python.builtin.BaseException <- function(x, ...) {
    NextMethod()
    r_attr_nms <- grep("^r_", py_list_attributes(x), value = TRUE)
    if (length(r_attr_nms)) {
      r_attrs <- lapply(r_attr_nms,
                        function(nm)
                          as_r_value(py_get_attr(x, nm)))
      names(r_attrs) <- r_attr_nms
      r_traceback <- r_attrs$r_traceback
      r_attrs$r_traceback <- NULL
      str(r_attrs, no.list = TRUE)
      if(!is.null(r_traceback)) {
        cat(" $ r_traceback: \n")
        traceback(r_traceback)
      }
    }
    invisible(x)
}

#' @export
`$.python.builtin.BaseException` <- function(x, name) {
    if ("condition" %in% .Class &&
        (identical(name, "call") || identical(name, "message"))) {
        # warning("Please use conditionCall() or conditionMessage() instead of $call or $message")
        return(switch(name,
            call = conditionCall(x),
            message = conditionMessage(x)
        ))
    }
    py_get_attr(x, name, TRUE)
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
  on.exit(.globals$py_last_exception <- ex)
  tryCatch(
    py_eval("_", convert = FALSE),
    error = function(e) py_none()
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
  message(paste("+", shQuote(command), paste0(args, collapse = " ")))
  system2(command, args, ...)
}


rm_all_reticulate_state <- function() {
  unlink(rappdirs::user_data_dir("r-reticulate", NULL), recursive = TRUE, force = TRUE)
  unlink(rappdirs::user_data_dir("r-miniconda", NULL), recursive = TRUE, force = TRUE)
  unlink(rappdirs::user_data_dir("r-miniconda-arm64", NULL), recursive = TRUE, force = TRUE)
}
