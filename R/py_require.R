#' Declare Python requirements
#'
#' It allows you to specify the Python packages, and their versions, to use
#' during your working session. It also allows to specify Python version
#' requirements. It uses [uv](https://docs.astral.sh/uv/) to automatically
#' resolves multiple version requirements of the same package (e.g.:
#' 'numpy>=2.2.0', numpy==2.2.2'), as well as resolve multiple Python version
#' requirements (e.g.: '>=3.10', '3.11').  `uv` will automatically download and
#' install the resulting Python version and packages, so there is no need to
#' take any steps prior to starting the Python session.
#'
#'
#' The virtual environment will not be initialized until the users attempts to
#' interacts with Python for the first time during the session. Typically,
#' that would be the first time `import()` is called.
#'
#' If `uv` is not installed, `reticulate` will attempt to download and install
#' a version of it in an isolated folder. This will allow you to get the
#' advantages of `uv`, without modifying your computer's environment.
#'
#'
#' @param packages A vector of Python packages to make available during the
#' working session.
#'
#' @param python_version A vector of one, or multiple, Python versions to
#' consider. `uv` will not be able to process conflicting Python versions
#' (e.g.: '>=3.11', '3.10').
#'
#' @param action What `py_require()` should do with the packages and Python
#' version provided during the given command call. There are three options:
#' - add - Adds the requirement to the list
#' - remove - Removes the requirement form the list. It has to be an exact match
#' to an existing requirement. For example, if 'numpy==2.2.2' is currently on
#' the list, passing 'numpy' with a 'remove' action will affect the list.
#' - set - Deletes any requirement already defined, and replaces them with what
#' is provided in the command call. Packages and Python version can be
#' independently set.
#'
#' @param exclude_newer Leverages a feature from `uv` that allows you to limit
#' the candidate package versions to those that were uploaded prior to a given
#' date. During the working session, the date can be "added" only one time.
#' After the first time the argument is used, only the 'set' `action` can
#' override the date afterwards.
#'
#' @export
py_require <- function(packages = NULL,
                       python_version = NULL,
                       exclude_newer = NULL,
                       action = c("add", "remove", "set")) {
  action <- match.arg(action)

  uv_initialized <- is_python_initialized() &&
    is_uv_environment(dirname(dirname(py_exe())))

  if (uv_initialized && !is.null(python_version)) {
    stop(
      "Python version requirements cannot be ",
      "changed after Python has been initialized"
    )
  }

  if (!is.null(exclude_newer) &&
    action != "set" &&
    !is.null(get_python_reqs("exclude_newer"))
  ) {
    stop(
      "`exclude_newer` is already set to '",
      get_python_reqs("exclude_newer"),
      "', use `action` 'set' to override"
    )
  }

  if (missing(packages) && missing(python_version) && missing(exclude_newer)) {
    return(get_python_reqs())
  }

  req_packages <- get_python_reqs("packages")
  if (!is.null(packages)) {
    req_packages <- switch(action,
      add = unique(c(req_packages, packages)),
      remove = setdiff(req_packages, packages),
      set = packages
    )
  }

  req_python <- get_python_reqs("python_version")
  if (!is.null(python_version)) {
    req_python <- switch(action,
      add = unique(c(python_version, req_python)),
      remove = setdiff(req_python, python_version),
      set = python_version
    )
  }

  top_env <- topenv(parent.frame())

  pr <- .globals$python_requirements
  pr$packages <- req_packages
  pr$python_version <- req_python
  pr$exclude_newer <- pr$exclude_newer %||% exclude_newer
  pr$history <- c(pr$history, list(list(
    requested_from = environmentName(top_env),
    env_is_package = isNamespace(top_env),
    packages = packages,
    python_version = python_version,
    exclude_newer = exclude_newer,
    action = action
  )))

  if (uv_initialized) {
    new_path <- get_or_create_venv(
      packages = pr$packages,
      python_version = pr$python_version,
      exclude_newer = pr$exclude_newer
    )
    new_venv_path(new_path)
  }

  .globals$python_requirements <- pr

  invisible()
}

# Print ------------------------------------------------------------------------

#' @export
print.python_requirements <- function(x, ...) {
  packages <- x$packages
  if (is.null(packages)) {
    packages <- "[No package(s) specified]"
  }
  python_version <- x$python_version
  if (is.null(python_version)) {
    python_version <- "[No Python version specified]"
  }
  cat(paste0(rep("=", 26), collapse = ""))
  cat(" Python requirements ")
  cat(paste0(rep("=", 26), collapse = ""), "\n")
  cat(requirements_print(
    packages = packages,
    python_version = python_version,
    exclude_newer = x$exclude_newer
  ))
  cat("\n")

  # TODO - Add support for "remove" action (will require a full parsing of history)
  requested_from <- as.character(lapply(x$history, function(x) x$requested_from))
  history <- x$history[requested_from != "R_GlobalEnv"]
  is_package <- as.logical(lapply(history, function(x) x$env_is_package))
  if (any(is_package)) {
    cat("-- R package requests ")
    cat(paste0(rep("-", 51), collapse = ""), "\n")
    requirements_table(history[is_package], "R package")
  }

  if (any(!is_package)) {
    cat("-- Environment requests ")
    cat(paste0(rep("-", 49), collapse = ""), "\n")
    requirements_table(history[is_package], "R package")
  }

  return(invisible())
}


pad_length <- function(x = "", len) {
  padding <- paste0(rep(" ", len - nchar(x)), collapse = "")
  paste0(x, padding)
}

requirements_table <- function(history, from_label) {
  console_width <- 73
  python_width <- 20
  requested_from <- as.character(lapply(history, function(x) x$requested_from))
  pkg_names <- c(unique(requested_from), from_label)
  name_width <- max(nchar(pkg_names)) + 1
  pkg_width <- console_width - python_width - name_width
  header <- list(list(
    requested_from = from_label,
    packages = "Python package(s)",
    python_version = "Python version"
  ))
  history <- c(header, history)
  for (pkg_entry in history) {
    pkg_lines <- strwrap(
      x = paste0(pkg_entry$packages, collapse = ", "),
      width = pkg_width
    )
    python_lines <- strwrap(
      x = paste0(pkg_entry$python_version, collapse = ", "),
      width = python_width
    )
    max_lines <- max(c(length(python_lines), length(pkg_lines)))
    for (i in seq_len(max_lines)) {
      nm <- ifelse(i == 1, pkg_entry$requested_from, "")
      pk <- ifelse(i <= length(pkg_lines), pkg_lines[i], "")
      py <- ifelse(i <= length(python_lines), python_lines[i], "")
      cat(pad_length(nm, name_width))
      cat(pad_length(pk, pkg_width))
      cat(pad_length(py, python_width))
      cat("\n")
    }
  }
}

# Get requirements ---------------------------------------------------------

get_python_reqs <- function(x = NULL) {
  pr <- .globals$python_requirements
  if (is.null(pr)) {
    pr <- structure(
      .Data = list(
        python_version = c(),
        packages = c(),
        exclude_newer = NULL,
        history = list()
      ),
      class = "python_requirements"
    )
    pkg_prime <- "numpy"
    pr$packages <- pkg_prime
    pr$history <- list(list(
      requested_from = "reticulate",
      env_is_package = TRUE,
      action = "add",
      packages = pkg_prime
    ))
    .globals$python_requirements <- pr
  }
  if (!is.null(x)) {
    pr[[x]]
  } else {
    pr
  }
}

# uv ---------------------------------------------------------------------------

uv_binary <- function() {
  uv <- Sys.getenv("RETICULATE_UV", NA)
  if (!is.na(uv)) {
    return(path.expand(uv))
  }

  uv <- getOption("reticulate.uv_binary")
  if (!is.null(uv)) {
    return(path.expand(uv))
  }

  uv <- as.character(Sys.which("uv"))
  if (uv != "") {
    return(path.expand(uv))
  }

  uv <- path.expand("~/.local/bin/uv")
  if (file.exists(uv)) {
    return(path.expand(uv))
  }

  uv <- file.path(rappdirs::user_cache_dir("r-reticulate", NULL), "bin", "uv")
  if (file.exists(uv)) {
    return(path.expand(uv))
  }

  if (is_windows()) {

  } else if (is_macos() || is_linux()) {
    install_uv.sh <- tempfile("install-uv-", fileext = ".sh")
    res <-tryCatch(
      download.file("https://astral.sh/uv/install.sh", install_uv.sh, quiet = TRUE),
      warning = function(x) NULL,
      error = function(x) NULL
    )
    if(is.null(res)) {
      return(NULL)
    }
    Sys.chmod(install_uv.sh, mode = "0755")
    dir.create(dirname(uv), showWarnings = FALSE)
    # https://github.com/astral-sh/uv/blob/main/docs/configuration/installer.md
    system2(install_uv.sh, c("--quiet"), env = c(
      "INSTALLER_NO_MODIFY_PATH=1", paste0("UV_INSTALL_DIR=", maybe_shQuote(dirname(uv)))
    ))
    return(path.expand(uv))
  }
}

# TODO: we should pass --cache-dir=file.path(rappdirs::user_cache_dir("r-reticulate"), "uv-cache")
# if we are using a reticulate-managed uv installation.

get_or_create_venv <- function(packages = get_python_reqs("packages"),
                               python_version = get_python_reqs("python_version"),
                               exclude_newer = get_python_reqs("exclude_newer")) {

  uv_binary_path <- uv_binary()

  if(is.null(uv_binary_path)) {
    return(NULL)
  }

  if (length(packages)) {
    pkg_arg <- as.vector(rbind("--with", maybe_processx(packages)))
  } else {
    pkg_arg <- NULL
  }

  if (length(python_version)) {
    has_const <- substr(python_version, 1, 1) %in% c(">", "<", "=", "!")
    python_version[!has_const] <- paste0("==", python_version[!has_const])
    python_arg <- c("--python", paste0(maybe_processx(python_version), collapse = ","))
  } else {
    python_arg <- NULL
  }

  if (!is.null(exclude_newer)) {
    # todo, accept a POSIXct/lt, format correctly
    exclude_arg <- c("--exclude-newer", maybe_shQuote(exclude_newer))
  } else {
    exclude_arg <- NULL
  }

  command_arg <- "import sys; print(sys.executable);"
  if (!will_use_processx()) {
    command_arg <- maybe_shQuote(command_arg)
  }

  args <- c(
    "run",
    "--color", "never",
    "--no-project",
    "--python-preference=only-managed",
    exclude_arg,
    python_arg,
    pkg_arg,
    "python", "-c", command_arg
  )

  if (will_use_processx()) {
    on.exit(
      try(p$kill(), silent = TRUE),
      add = TRUE
    )
    p <- processx::process$new(
      command = uv_binary_path,
      args = args,
      stderr = "|",
      stdout = "|"
    )
    sp <- cli::make_spinner(template = "Downloading Python dependencies {spin}")
    repeat {
      pr <- p$poll_io(100)
      if (all(as.vector(pr[c("error", "output")]) == "ready")) break
      sp$spin()
    }
    sp$finish()
    cmd_err <- p$read_error()
    cmd_out <- p$read_output()
    cmd_failed <- identical(cmd_out, "")
  } else {
    result <- suppressWarnings(system2(
      command = uv_binary(),
      args = args,
      stderr = TRUE,
      stdout = TRUE
    ))
    cmd_failed <- !is.null(attributes(result))
    if (cmd_failed) {
      cmd_err <- paste0(result, collapse = "\n")
    } else {
      cmd_err <- paste0(result[[1]], "\n")
      cmd_out <- result[[2]]
    }
  }

  if (cmd_failed) {
    writeLines(cmd_err, con = stderr())
    msg <- requirements_print(
      packages = packages,
      python_version = python_version,
      exclude_newer = exclude_newer
    )
    msg <- c(msg, paste0(rep("-", 73), collapse = ""))
    writeLines(msg, con = stderr())
    stop(
      "Call `py_require()` to remove or replace conflicting requirements.",
      call. = FALSE
    )
  }
  cat(cmd_err)
  capture.output(cat(cmd_out))
}

will_use_processx <- function() {
  interactive() && !isatty(stderr()) &&
    (is_rstudio() || is_positron()) &&
    requireNamespace("cli") && requireNamespace("processx")
}

maybe_processx <- function(x) {
  if (!will_use_processx()) {
    x <- maybe_shQuote(x)
  }
  x
}

requirements_print <- function(packages = NULL,
                               python_version = NULL,
                               exclude_newer = NULL) {
  msg <- c(
    paste0(
      "-- Current requirements ", paste0(rep("-", 49), collapse = ""),
      collapse = ""
    ),
    if (!is.null(python_version)) {
      paste0(" Python:   ", paste0(python_version, collapse = ", "))
    },
    if (!is.null(packages)) {
      # TODO: wrap+indent+un_shQuote python packages
      pkg_lines <- strwrap(paste0(packages, collapse = ", "), 60)
      pkg_col <- c(" Packages: ", rep("           ", length(pkg_lines) - 1))
      out <- NULL
      for (i in seq_along(pkg_lines)) {
        out <- c(out, paste0(pkg_col[[i]], pkg_lines[[i]]))
      }
      out
    },
    if (!is.null(exclude_newer)) {
      paste0(" Exclude:  Anything newer than ", exclude_newer)
    }
  )
  paste0(msg, collapse = "\n")
}

new_venv_path <- function(path) {
  new_config <- python_config(path)
  if (new_config$libpython == .globals$py_config$libpython) {
    py_activate_virtualenv(file.path(dirname(path), "activate_this.py"))
    .globals$py_config <- new_config
    .globals$py_config$available <- TRUE
  } else {
    # TODO: Better error message?
    stop("New environment does not use the same Python binary")
  }
  invisible()
}

is_uv_environment <- function(dir) {
  cfg_file <- file.path(dir, "pyvenv.cfg")
  if (file.exists(cfg_file)) {
    cfg <- readLines(cfg_file)
    return(any(grepl("uv = ", cfg)))
  } else {
    return(FALSE)
  }
}
