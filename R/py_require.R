#' Declare Python Requirements
#'
#' `py_require()` allows you to declare Python requirements for the R session,
#' including Python packages, any version constraints on those packages, and any
#' version constraints on Python itself. Reticulate can then automatically
#' create and use an ephemeral Python environment that satisfies all these
#' requirements.
#'
#' Reticulate will only use an ephemeral environment if no other Python
#' installation is found earlier in the [Order of
#' Discovery](https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery).
#' You can also force reticulate to use an ephemeral environment by setting
#' `Sys.setenv(RETICULATE_USE_MANAGED_VENV = "yes")`.
#'
#' The ephemeral virtual environment is not created until the user interacts
#' with Python for the first time in the R session, typically when `import()` is
#' first called.
#'
#' If `py_require()` is called with new requirements after reticulate has
#' already initialized an ephemeral Python environment, a new ephemeral
#' environment is activated on top of the existing one. Once Python is
#' initialized, only adding packages is supported---removing packages, changing
#' the Python version, or modifying `exclude_newer` is not possible.
#'
#' Calling `py_require()` without arguments returns a list of the currently
#' declared requirements.
#'
#' R packages can also call `py_require()` (e.g., in `.onLoad()` or elsewhere)
#' to declare Python dependencies. The print method for `py_require()` displays
#' the Python dependencies declared by R packages in the current session.
#'
#' @note Reticulate uses [`uv`](https://docs.astral.sh/uv/) to resolve Python
#'   dependencies. Many `uv` options can be customized via environment
#'   variables, as described
#'   [here](https://docs.astral.sh/uv/configuration/environment/). For example:
#'   - If temporarily offline, set `Sys.setenv(UV_OFFLINE=1)`.
#'   - To use a different index: `Sys.setenv(UV_INDEX = "https://download.pytorch.org/whl/cpu")`.
#'   - To allow resolving a prerelease dependency: `Sys.setenv(UV_PRERELEASE="allow")`.
#'
#' @param packages A character vector of Python packages to be available during
#'   the session. These can be simple package names like `"jax"` or names with
#'   version constraints like `"jax[cpu]>=0.5"`.
#'
#' @param python_version A character vector of Python version constraints \cr
#'   (e.g., `"3.10"` or `">=3.9,<3.13,!=3.11"`).
#'
#' @param ... Reserved for future extensions; must be empty.
#'
#' @param action Determines how `py_require()` processes the provided
#'   requirements. Options are:
#'   - `add`: Adds the entries to the current set of requirements.
#'   - `remove`: Removes__exact_ matches from the requirements list. For example,
#'   if `"numpy==2.2.2"` is in the list, passing `"numpy"` with `action =
#'   "remove"` will not remove it. Requests to remove nonexistent entries are
#'   ignored.
#'   - `set`: Clears all existing requirements and replaces them with the
#'   provided ones. Packages and the Python version can be set independently.
#'
#' @param exclude_newer Restricts package versions to those published before a
#'   specified date. This offers a lightweight alternative to freezing package
#'   versions, helping guard against Python package updates that break a
#'   workflow. Once `exclude_newer` is set, only the `set` action can override
#'   it.
#'
#' @export
py_require <- function(packages = NULL,
                       python_version = NULL,
                       ...,
                       exclude_newer = NULL,
                       action = c("add", "remove", "set")) {

  if (length(list(...)))
    stop("... must be empty")

  pr <- py_reqs_get()

  if (missing(packages) && missing(python_version) && missing(exclude_newer)) {
    return(pr)
  }

  action <- match.arg(action)
  called_from_package <- isNamespace(topenv(parent.frame()))
  uv_initialized <- is_python_initialized() && is_ephemeral_reticulate_uv_env(py_exe())

  # TODO: called_from_package_onLoad <- in_onload()
  signal_and_exit <- if (called_from_package) warn_and_return else stop

  if (!is.null(python_version)) {
    python_version <- unlist(strsplit(python_version, ",", fixed = TRUE))

    if (uv_initialized) {

      current_py_version <- py_version(patch = TRUE)
      for (check in as_version_constraint_checkers(python_version)) {
        if (!isTRUE(check(current_py_version))) {
          signal_and_exit(paste0(collapse = "",
            "Python version requirements cannot be ",
            "changed after Python has been initialized.\n",
            "* Python version request: '", python_version, "'",
            if (called_from_package) paste0(" (from package:", parent.pkg(), ")"),
            "\n",
            "* Python version initialized: '", as.character(current_py_version), "'"
          ))
        }
      }

    } else {

      pr$python_version <- py_reqs_action(action,
                                          python_version,
                                          py_reqs_get("python_version"))

    }

  }

  if (!is.null(exclude_newer)) {
    if (called_from_package) {
      stop("`exclude_newer` cannot be set inside a package")
    }

    if (uv_initialized) {

      if (!identical(exclude_newer, pr$exclude_newer))
        stop("`exclude_newer` cannot be changed after Python has initialized.")

    } else {

      switch(action,
        add = {
          if (!is.null(pr$exclude_newer)) {
            # TODO: we can check if the new request is already satisfied
            # by the old request. e.g.,
            #   as.POSIXct(exclude_newer) >= as.POSIXct(pr$exclude_newer)
            stop(
              "`exclude_newer` is already set to '",
              py_reqs_get("exclude_newer"),
              "', use `action = 'set'` to override"
            )
          }
        },
        remove = {
          if (identical(exclude_newer, pr$exclude_newer)) {
            exclude_newer <- NA
          }
        },
        set = {}
      )

      if (is.na(exclude_newer) || identical(exclude_newer, "")) {
        # NA or "" are the sentinel for removing exclude_newer
        # (since NULL sentinel already is taken)
        exclude_newer <- NULL
      }

      pr$exclude_newer <- exclude_newer
    }
  }

  if (!is.null(packages)) {
    if (uv_initialized) {
      switch(action,
        add = {
          if(all(packages %in% pr$packages)) {
            packages <- NULL # no-op, skip activating new env
          } else {
            bare_name <- function(x) sub("^([^[!=><]+).*", "\\1", x)
            if (any(bare_name(packages) %in% bare_name(pr$packages))) {
              # e.g., if user calls 'numpy<2' after already initialized with 'numpy>2'
              signal_and_exit("After Python has initialized, only `action = 'add'` with new packages is supported.")
              packages <- NULL
            }
            pr$packages <- unique(c(packages, pr$packages))
          }
        },
        remove = {
          if (any(packages %in% pr$packages))
            signal_and_exit("After Python has initialized, only `action = 'add'` is supported.")
        },
        set = {
          if (!base::setequal(packages, pr$packages))
            signal_and_exit("After Python has initialized, only `action = 'add'` is supported.")
        })
    } else {
      pr$packages <- py_reqs_action(action, packages, py_reqs_get("packages"))
    }
  }

  if (uv_initialized && action == "add" && !is.null(packages)) {
    tryCatch({
      new_path <- uv_get_or_create_env()
      new_config <- python_config(new_path)
      if (new_config$libpython == .globals$py_config$libpython) {
        py_activate_virtualenv(file.path(dirname(new_path), "activate_this.py"))
        .globals$py_config <- new_config
        .globals$py_config$available <- TRUE
        # TODO: sync os.environ with R Sys.getenv()?
      } else {
        # TODO: Better error message?
        stop("New environment does not use the same Python binary")
      }
    }, error = signal_and_exit)
  }

  pr$history <- c(pr$history, list(list(
    requested_from = environmentName(topenv(parent.frame())),
    env_is_package = called_from_package,
    packages = packages,
    python_version = python_version,
    exclude_newer = exclude_newer,
    action = action
  )))
  .globals$python_requirements <- pr


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
    python_version <- paste0("[No Python version specified. Will default to '", resolve_python_version() , "']")
  }

  requested_from <- as.character(lapply(x$history, function(x) x$requested_from))
  history <- x$history[requested_from != "R_GlobalEnv"]
  is_package <- as.logical(lapply(history, function(x) x$env_is_package))

  if (requireNamespace("cli", quietly = TRUE)) {
    withr::with_options(
      list("cli.width" = 73),
      {
        cli::cli_div(
          theme = list(rule = list(color = "cyan", "line-type" = "double"))
        )
        cli::cli_rule(center = "Python requirements")
        cli::cli_div(
          theme = list(rule = list("line-type" = "single"))
        )
        cli::cli_rule("Current requirements")
        cat(py_reqs_format(
          packages = packages,
          python_version = python_version,
          exclude_newer = x$exclude_newer,
          use_cli = TRUE
        ))
        cat("\n")
        if (any(is_package)) {
          cli::cli_rule("R package requests")
          py_reqs_table(history[is_package], "R package", use_cli = TRUE)
        }
        if (any(!is_package)) {
          cli::cli_rule("Environment requests")
          py_reqs_table(history[!is_package], "R package", use_cli = TRUE)
        }
      }
    )
  } else {
    cat(paste0(rep("=", 26), collapse = ""))
    cat(" Python requirements ")
    cat(paste0(rep("=", 26), collapse = ""), "\n")
    cat(py_reqs_format(
      packages = packages,
      python_version = python_version,
      exclude_newer = x$exclude_newer
    ))
    cat("\n")
    if (any(is_package)) {
      cat("-- R package requests ")
      cat(paste0(rep("-", 51), collapse = ""), "\n")
      py_reqs_table(history[is_package], "R package")
    }
    if (any(!is_package)) {
      cat("-- Environment requests ")
      cat(paste0(rep("-", 49), collapse = ""), "\n")
      py_reqs_table(history[!is_package], "R package")
    }
  }
  invisible()
}

# Python requirements - utils --------------------------------------------------

py_reqs_pad <- function(x = "", len, use_cli, is_title = FALSE) {
  padding <- paste0(rep(" ", len - nchar(x)), collapse = "")
  ret <- paste0(x, padding)
  if (use_cli) {
    if (is_title) {
      ret <- cli::col_blue(ret)
    } else {
      ret <- cli::col_grey(ret)
    }
  }
  ret
}

py_reqs_table <- function(history, from_label, use_cli = FALSE) {
  console_width <- 73
  python_width <- 20
  requested_from <- as.character(lapply(history, function(x) x$requested_from))
  pkg_names <- c(unique(requested_from), from_label)
  name_width <- max(nchar(pkg_names)) + 1
  pkg_width <- console_width - python_width - name_width
  header <- list(list(
    requested_from = from_label,
    packages = "Python packages",
    python_version = "Python version",
    is_title = 1
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
      cat(py_reqs_pad(nm, name_width, use_cli, !is.null(pkg_entry$is_title)))
      cat(py_reqs_pad(pk, pkg_width, use_cli, !is.null(pkg_entry$is_title)))
      cat(py_reqs_pad(py, python_width, use_cli, !is.null(pkg_entry$is_title)))
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

py_reqs_format <- function(packages = NULL,
                          python_version = NULL,
                          exclude_newer = NULL,
                          use_cli = FALSE) {
  msg <- c(
    if (!use_cli) {
      paste0(
        "-- Current requirements ", paste0(rep("-", 49), collapse = ""),
        collapse = ""
      )
    },
    if (!is.null(python_version)) {
      python <- ifelse(use_cli, cli::col_blue("Python:"), "Python:")
      python_version <- paste0(python_version, collapse = ", ")
      python_version <- ifelse(use_cli, cli::col_grey(python_version), python_version)
      paste0(" ", python, "   ", python_version)
    },
    if (!is.null(packages)) {
      pkg_lines <- strwrap(paste0(packages, collapse = ", "), 60)
      pkgs <- "Packages:"
      if (use_cli) {
        pkgs <- cli::col_blue(pkgs)
        pkg_lines <- as.character(lapply(pkg_lines, cli::col_grey))
      }
      pkg_col <- c(paste0(" ", pkgs, " "), rep("           ", length(pkg_lines) - 1))
      out <- NULL
      for (i in seq_along(pkg_lines)) {
        out <- c(out, paste0(pkg_col[[i]], pkg_lines[[i]]))
      }
      out
    },
    if (!is.null(exclude_newer)) {
      exclude <- ifelse(use_cli, cli::col_blue("Exclude:"), "Exclude:")
      exclude_newer <- paste0("  Anything newer than ", exclude_newer)
      exclude_newer <- ifelse(use_cli, cli::col_grey(exclude_newer), exclude_newer)
      paste0(" ", exclude, exclude_newer)
    }
  )
  paste0(msg, collapse = "\n")
}

py_reqs_get <- function(x = NULL) {
  pr <- .globals$python_requirements
  if (is.null(pr)) {
    pr <- structure(
      list(
        python_version = NULL,
        packages = NULL,
        exclude_newer = NULL,
        history = list()
      ),
      class = "python_requirements"
    )
    pkg_prime <- c("numpy", if (is_positron()) "ipykernel")
    pr$packages <- pkg_prime
    pr$history <- list(list(
      requested_from = "reticulate",
      env_is_package = TRUE,
      action = "add",
      packages = pkg_prime
    ))
    .globals$python_requirements <- pr
  }
  if (is.null(x)) {
    return(pr)
  }
  if (x == "python_version") {
    uv_initialized <- is_python_initialized() && is_ephemeral_reticulate_uv_env(py_exe())
    if (uv_initialized)
      return(as.character(py_version(TRUE)))
  }
  pr[[x]]
}

# uv ---------------------------------------------------------------------------

uv_binary <- function(bootstrap_install = TRUE) {
  required_version <- numeric_version("0.6.1")
  is_usable_uv <- function(uv) {
    if (is.null(uv) || is.na(uv) || uv == "" || !file.exists(uv)) {
      return(FALSE)
    }
    ver <- suppressWarnings(system2(uv, "--version", stderr = TRUE, stdout = TRUE))
    if (!is.null(attr(ver, "status"))) {
      return(FALSE)
    }
    ver <- numeric_version(sub("uv ([0-9.]+) .*", "\\1", ver), strict = FALSE)
    !is.na(ver) && ver >= required_version
  }

  uv <- Sys.getenv("RETICULATE_UV", NA)
  if (is_usable_uv(uv)) {
    return(path.expand(uv))
  }

  uv <- getOption("reticulate.uv_binary")
  if (is_usable_uv(uv)) {
    return(path.expand(uv))
  }

  uv <- as.character(Sys.which("uv"))
  if (is_usable_uv(uv)) {
    return(path.expand(uv))
  }

  uv <- path.expand("~/.local/bin/uv")
  if (is_usable_uv(uv)) {
    return(path.expand(uv))
  }

  uv <- path.expand(file.path(
    rappdirs::user_cache_dir("r-reticulate", NULL),
    "bin", if (is_windows()) "uv.exe" else "uv"
  ))
  if (file.exists(uv)) {
    if (!is_usable_uv(uv)) # exists, but version too old
      system2(uv, "self update")
    return(uv)
  }

  if (bootstrap_install) {
    # Install 'uv' in the 'r-reticulate' sub-folder inside the user's cache directory
    # https://github.com/astral-sh/uv/blob/main/docs/configuration/installer.md
    file_ext <- if (is_windows()) ".ps1" else ".sh"
    url <- paste0("https://astral.sh/uv/install", file_ext)
    install_uv <- tempfile("install-uv-", fileext = file_ext)
    download.file(url, install_uv, quiet = TRUE)
    if (!file.exists(install_uv)) {
      return(NULL)
      # stop("Unable to download Python dependencies. Please install `uv` manually.")
    }

    if (is_windows()) {
      system2("powershell", c(
        "-ExecutionPolicy",
        "ByPass",
        "-c",
        paste0(
          "$env:UV_INSTALL_DIR='", dirname(uv), "';",
          "$env:INSTALLER_NO_MODIFY_PATH= 1;",
          # 'Out-Null' makes installation silent
          "irm ", install_uv, " | iex *> Out-Null"
        )
      ))
    } else if (is_macos() || is_linux()) {
      Sys.chmod(install_uv, mode = "0755")
      dir.create(dirname(uv), showWarnings = FALSE, recursive = TRUE)
      system2(install_uv, c("--quiet"),
        env = c(
          "INSTALLER_NO_MODIFY_PATH=1",
          paste0("UV_INSTALL_DIR=", maybe_shQuote(dirname(uv)))
        )
      )
    }
  }

  if (file.exists(uv)) uv else NULL # print visible
}

uv_get_or_create_env <- function(packages = py_reqs_get("packages"),
                                 python_version = py_reqs_get("python_version"),
                                 exclude_newer = py_reqs_get("exclude_newer")) {

  uv <- uv_binary() %||% return() # error?

  resolved_python_version <- resolve_python_version(constraints = python_version, uv = uv)

  # capture args; maybe used in error message later
  call_args <- list(
    packages = packages,
    python_version = python_version %||%
      paste(resolved_python_version, "(reticulate default)"),
    exclude_newer = exclude_newer
  )

  if (length(packages))
    packages <- as.vector(rbind("--with", packages))

  python_version <- c("--python", resolved_python_version)

  if (!is.null(exclude_newer)) {
    # todo, accept a POSIXct/lt, format correctly
    exclude_newer <- c("--exclude-newer", exclude_newer)
  }

  # TODO?: use default uv cache if using user-installed uv?
  # need to refactor detecting approach in py_install() and py_require()
  cache_dir <- #if (is_reticulate_managed_uv(uv))
    c("--cache-dir", reticulate_managed_uv_cache_dir())

  withr::local_envvar(c(
    VIRTUAL_ENV = NA,
    if (is_positron())
      c(RUST_LOG = NA)
  ))

  uv_args <- c(
    "run",
    "--no-project",
    "--python-preference", "only-managed",
    cache_dir,
    python_version,
    exclude_newer,
    packages,
    "--",
    "python", "-c", "import sys; print(sys.executable);"
  )

  # debug print system call
  if (Sys.getenv("_RETICULATE_DEBUG_UV_") == "1")
    message(paste0(c(shQuote(uv), maybe_shQuote(uv_args)), collapse = " "))

  env_python <- suppressWarnings(system2(uv, maybe_shQuote(uv_args), stdout = TRUE))
  error_code <- attr(env_python, "status", TRUE)

  if (!is.null(error_code)) {
    cat("uv error code: ", error_code, "\n", sep = "", file = stderr())
    msg <- do.call(py_reqs_format, call_args)
    writeLines(c(msg, strrep("-", 73L)), con = stderr())
    if (error_code == 2) {
      cat(
        "Hint: If you are temporarily offline, try setting `Sys.setenv(UV_OFFLINE=1)`.\n",
        file = stderr()
      )
    }
    stop("Call `py_require()` to remove or replace conflicting requirements.")
  }

  env_python
}

#' uv run tool
#'
#' Run a Command Line Tool distributed as a Python package. Packages are automatically
#' download and installed into a cached, ephemeral, and isolated environment on the first run.
#'
#' @param tool,args A character vector of command and arguments. Arguments are
#'   not quoted for the shell, so you may need to use [`shQuote()`].
#' @param from Use the given python package to provide the command.
#' @param with Run with the given Python packages installed. You can also
#'   specify version constraints like `"ruff>=0.3.0"`.
#' @param python_version A python version string, or character vector of python
#'   version constraints.
#'
#' @inheritDotParams base::system2 -command
#'
#' @details
#' ## Examples
#' ```r
#' uv_run_tool("pycowsay", shQuote("hello from reticulate"))
#' uv_run_tool("markitdown", shQuote(file.path(R.home("doc"), "NEWS.pdf")), stdout = TRUE)
#' uv_run_tool("kaggle competitions download -c dogs-vs-cats")
#' uv_run_tool("ruff", "--help")
#' uv_run_tool("ruff format", shQuote(Sys.glob("**.py")))
#' uv_run_tool("http", from = "httpie")
#' uv_run_tool("http", "--version", from = "httpie<3.2.4", stdout = TRUE)
#' uv_run_tool("saved_model_cli", "--help", from = "tensorflow")
#' ```
#' @seealso <https://docs.astral.sh/uv/guides/tools/>
#' @returns Return value of [`system2()`]
#' @export
#' @md
uv_run_tool <- function(tool, args = character(), ..., from = NULL, with = NULL, python_version = NULL) {
  system2(uv_binary(), c(
    "tool",
    "run",
    "--isolated",
    "--python-preference=only-managed",
    "--python", resolve_python_version(constraints = python_version),
    if (length(from)) c("--from", maybe_shQuote(from)),
    if (length(with)) c(rbind("--with", maybe_shQuote(with))),
    "--",
    tool,
    args
  ), ...)
}


# uv - utils -------------------------------------------------------------------


is_reticulate_managed_uv <- function(uv = uv_binary(bootstrap_install = FALSE)) {
  if(is.null(uv) || !file.exists(uv)) {
    # no user-installed uv - uv will be bootstrapped by reticulate
    return(TRUE)
  }

  managed_uv_path <- path.expand(file.path(
    rappdirs::user_cache_dir("r-reticulate", NULL),
    "bin", if (is_windows()) "uv.exe" else "uv"
  ))

  uv == managed_uv_path
}

is_ephemeral_reticulate_uv_env <- function(path) {
  startsWith(path, reticulate_managed_uv_cache_dir())
}

reticulate_managed_uv_cache_dir <- function() {
  path.expand(file.path(
    rappdirs::user_cache_dir("r-reticulate", NULL),
    "uv-cache"
  ))
}

uv_cache_dir <- function(uv = uv_binary(bootstrap_install = FALSE)) {
  if (is_reticulate_managed_uv(uv)) {
    return(reticulate_managed_uv_cache_dir())
  }
  tryCatch(
    system2(uv, "cache dir",
            stdout = TRUE, stderr = FALSE,
            env = "NO_COLOR=1"),
    warning = function(w) NULL,
    error = function(e) NULL
  )
}


uv_python_list <- function(uv = uv_binary()) {
  x <- system2(uv, c("python list",
    "--python-preference only-managed",
    "--only-downloads",
    "--color never",
    "--output-format json"
    ),
    stdout = TRUE
  )

  x <- jsonlite::parse_json(x)
  x <- unlist(lapply(x, `[[`, "version"))

  # to parse default `--output-format text`
  # x <- grep("^cpython-", x, value = TRUE)
  # x <- sub("^cpython-([^-]+)-.*", "\\1", x)

  xv <- numeric_version(x, strict = FALSE)
  latest_minor_patch <- !duplicated(xv[, -3L]) & !is.na(xv)
  x <- x[order(latest_minor_patch, xv, decreasing = TRUE)]
  x
}

resolve_python_version <- function(constraints = NULL, uv = uv_binary()) {
  constraints <- as.character(constraints %||% "")
  constraints <- trimws(unlist(strsplit(constraints, ",", fixed = TRUE)))
  constraints <- constraints[nzchar(constraints)]

  if (length(constraints) == 0) {
    return(as.character(uv_python_list()[3L])) # default
  }

  # reflect a direct version specification like "3.11" or "3.14.0a3"
  if (length(constraints) == 1 && !substr(constraints, 1, 1) %in% c("=", ">", "<", "!")) {
    return(constraints)
  }

  # We perform custom constraint resolution to prefer slightly older Python releases.
  # uv tends to select the latest version, which often lack package support
  # See: https://devguide.python.org/versions/

  # Get latest patch for each minor version
  candidates <- uv_python_list(uv)
  # E.g., candidates might be:
  #  c("3.13.1", "3.12.8", "3.11.11", "3.10.16", "3.9.21", "3.8.20")

  # Reorder candidates to prefer stable versions over bleeding edge
  ord <- as.integer(c(3, 4, 2, 5, 1))
  ord <- union(ord, seq_along(candidates))
  candidates <- candidates[ord]

  # Maybe add non-latest patch levels to candidates if they're explicitly
  # mentioned in constraints
  append(candidates) <- sub("^[<>=!]{1,2}", "", constraints)

  candidates <- numeric_version(candidates, strict = FALSE)
  candidates <- candidates[!is.na(candidates)]

  for (check in as_version_constraint_checkers(constraints)) {
    satisfies_constraint <- check(candidates)
    candidates <- candidates[satisfies_constraint]
  }

  if (!length(candidates)) {
    constraints <- paste0(constraints, collapse = ",")
    msg <- paste0(
      'Requested Python version constraints could not be satisfied.\n',
      '  constraints: "', constraints, '"\n',
      'Hint: Call `py_require(python_version = <string>, action = "set")` to replace constraints.'
    )
    stop(msg)
  }

  as.character(candidates[1L])
}
