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
    is_uv_reticulate_managed_env(py_exe())

  if (uv_initialized && !is.null(python_version)) {
    stop(
      "Python version requirements cannot be ",
      "changed after Python has been initialized"
    )
  }

  if (!is.null(exclude_newer) &&
    action != "set" &&
    !is.null(py_reqs_get("exclude_newer"))
  ) {
    stop(
      "`exclude_newer` is already set to '",
      py_reqs_get("exclude_newer"),
      "', use `action` 'set' to override"
    )
  }

  if (missing(packages) && missing(python_version) && missing(exclude_newer)) {
    return(py_reqs_get())
  }

  pr <- py_reqs_get()
  pr$packages <- py_reqs_action(action, packages, py_reqs_get("packages"))
  pr$python_version <- py_reqs_action(action, python_version, py_reqs_get("python_version"))
  pr$exclude_newer <- pr$exclude_newer %||% exclude_newer
  pr$history <- c(pr$history, list(list(
    requested_from = environmentName(topenv(parent.frame())),
    env_is_package = isNamespace(topenv(parent.frame())),
    packages = packages,
    python_version = python_version,
    exclude_newer = exclude_newer,
    action = action
  )))
  .globals$python_requirements <- pr

  if (uv_initialized) {
    new_path <- uv_get_or_create_env()
    new_config <- python_config(new_path)
    if (new_config$libpython == .globals$py_config$libpython) {
      py_activate_virtualenv(file.path(dirname(new_path), "activate_this.py"))
      .globals$py_config <- new_config
      .globals$py_config$available <- TRUE
    } else {
      # TODO: Better error message?
      stop("New environment does not use the same Python binary")
    }
  }

  invisible()
}

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
  cat(py_reqs_print(
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
    py_reqs_table(history[is_package], "R package")
  }

  if (any(!is_package)) {
    cat("-- Environment requests ")
    cat(paste0(rep("-", 49), collapse = ""), "\n")
    py_reqs_table(history[is_package], "R package")
  }

  return(invisible())
}

# Python requirements - utils --------------------------------------------------

py_reqs_pad <- function(x = "", len) {
  padding <- paste0(rep(" ", len - nchar(x)), collapse = "")
  paste0(x, padding)
}

py_reqs_table <- function(history, from_label) {
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
  history <- lapply(unique(requested_from), py_reqs_flatten, history)
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
      cat(py_reqs_pad(nm, name_width))
      cat(py_reqs_pad(pk, pkg_width))
      cat(py_reqs_pad(py, python_width))
      cat("\n")
    }
  }
}


py_reqs_action <- function(action, x, y = NULL) {
  if (is.null(x)) {
    return(y)
  }
  switch(action,
    add = unique(c(y, x)),
    remove = setdiff(y, x),
    set = x
  )
}

py_reqs_flatten <- function(r_pkg = "", history) {
  req_packages <- NULL
  req_python <- NULL
  for (entry in history) {
    if (entry$requested_from == r_pkg | r_pkg == "") {
      req_packages <- py_reqs_action(entry$action, entry$packages, req_packages)
      req_python <- py_reqs_action(entry$action, entry$python_version, req_python)
    }
  }
  list(
    requested_from = r_pkg,
    packages = req_packages,
    python_version = req_python
  )
}

py_reqs_print <- function(packages = NULL,
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

py_reqs_get <- function(x = NULL) {
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
  uv_file <- ifelse(is_windows(), paste0(uv, ".exe"), uv)
  if (file.exists(uv_file)) {
    return(path.expand(uv))
  }

  # Installing 'uv' in the 'r-reticulate' sub-folder inside the user's
  # cache directory
  # https://github.com/astral-sh/uv/blob/main/docs/configuration/installer.md
  file_ext <- ifelse(is_windows(), ".ps1", ".sh")
  target_url <- paste0("https://astral.sh/uv/install", file_ext)
  install_uv <- tempfile("install-uv-", fileext = file_ext)
  res <- tryCatch(
    download.file(target_url, install_uv, quiet = TRUE),
    warning = function(x) NULL,
    error = function(x) NULL
  )
  if (is.null(res)) {
    return(NULL)
  }
  if (is_windows()) {
    system2(
      command = "powershell",
      args = c(
        "-ExecutionPolicy",
        "ByPass",
        "-c",
        paste0(
          "$env:UV_INSTALL_DIR='", dirname(uv), "';",
          "$env:INSTALLER_NO_MODIFY_PATH= 1;",
          # 'Out-Null' makes installation silent
          "irm ", install_uv, " | iex *> Out-Null"
        )
      )
    )
  } else if (is_macos() || is_linux()) {
    Sys.chmod(install_uv, mode = "0755")
    dir.create(dirname(uv), showWarnings = FALSE)
    system2(
      command = install_uv,
      args = c("--quiet"),
      env = c(
        "INSTALLER_NO_MODIFY_PATH=1",
        paste0("UV_INSTALL_DIR=", maybe_shQuote(dirname(uv)))
      )
    )
  }
  return(path.expand(uv))
}

uv_cache_dir <- function(...) {
  path <- file.path(rappdirs::user_cache_dir("r-reticulate", NULL), "uv-cache", ...)
  path.expand(path)
}

uv_get_or_create_env <- function(packages = py_reqs_get("packages"),
                                 python_version = py_reqs_get("python_version"),
                                 exclude_newer = py_reqs_get("exclude_newer")) {
  uv_binary_path <- uv_binary()

  if (is.null(uv_binary_path)) {
    return(NULL)
  }

  if (length(packages)) {
    pkg_arg <- as.vector(rbind("--with", uv_maybe_processx(packages)))
  } else {
    pkg_arg <- NULL
  }

  if (length(python_version)) {
    has_const <- substr(python_version, 1, 1) %in% c(">", "<", "=", "!")
    python_version[!has_const] <- paste0("==", python_version[!has_const])
    python_arg <- c("--python", paste0(uv_maybe_processx(python_version), collapse = ","))
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
  if (!uv_will_use_processx()) {
    command_arg <- maybe_shQuote(command_arg)
  }

  args <- c(
    "run",
    "--color", "never",
    "--no-project",
    "--cache-dir", uv_cache_dir(),
    "--python-preference=only-managed",
    exclude_arg,
    python_arg,
    pkg_arg,
    "python", "-c", command_arg
  )

  if (uv_will_use_processx()) {
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
    # This extra check is needed for Windows machines
    # p$read_error may come back empty, so using p$read_all_error
    # ensures forces the extraction. Running it as as the default
    # will make the process run slower on successful runs, which is
    # not ideal
    if (trimws(cmd_err) == "") {
      cmd_err <- p$read_all_error()
    }
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
    msg <- py_reqs_print(
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
  if (substr(cmd_out, nchar(cmd_out), nchar(cmd_out)) == "\n") {
    cmd_out <- substr(cmd_out, 1, nchar(cmd_out) - 1)
  }
  if (substr(cmd_out, nchar(cmd_out), nchar(cmd_out)) == "\r") {
    cmd_out <- substr(cmd_out, 1, nchar(cmd_out) - 1)
  }
  cmd_out
}

# uv - utils -------------------------------------------------------------------

uv_will_use_processx <- function() {
  interactive() && !isatty(stderr()) &&
    (is_rstudio() || is_positron()) &&
    requireNamespace("cli") && requireNamespace("processx")
}

uv_maybe_processx <- function(x) {
  if (!uv_will_use_processx()) {
    x <- maybe_shQuote(x)
  }
  x
}

is_uv_reticulate_managed_env <- function(dir) {
  str_cache <- as.character(uv_cache_dir())
  str_path <- as.character(dir)
  sub_path <- substr(str_path, 1, nchar(str_cache))
  str_cache == sub_path
}
