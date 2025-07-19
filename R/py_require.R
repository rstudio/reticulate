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
#' `Sys.setenv(RETICULATE_PYTHON="managed")`, or you can disable reticulate from
#' using an ephemeral environment by setting
#' `Sys.setenv(RETICULATE_USE_MANAGED_VENV="no")`.
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
#' @note
#'
#' Reticulate uses [`uv`](https://docs.astral.sh/uv/) to resolve Python
#' dependencies. Many `uv` options can be customized via environment variables,
#' as described [here](https://docs.astral.sh/uv/configuration/environment/).
#' For example:
#'   - If temporarily offline, to resolve packages from cache without checking for updates, set: \cr
#' `Sys.setenv(UV_OFFLINE = "1")`.
#'   - To use an additional package index: \cr
#' `Sys.setenv(UV_INDEX = "https://download.pytorch.org/whl/cpu")`. \cr (To add
#' multiple additional indexes, `UV_INDEX` can be a list of space-separated
#' urls).
#'   - To change the default package index: \cr
#' `Sys.setenv(UV_DEFAULT_INDEX = "https://my.org/python-packages-index/")`
#'   - To allow resolving a prerelease dependency: \cr
#' `Sys.setenv(UV_PRERELEASE = "allow")`.
#'   - To force `uv` to create ephemeral environments using the system python: \cr
#' `Sys.setenv(UV_PYTHON_PREFERENCE = "only-system")`
#'
#' For more advanced customization needs, there’s also the option to configure
#' `uv` with a user-level or system-level `uv.toml` file.
#'
#' ## Installing from alternate sources
#'
#' The `packages` argument also supports declaring a dependency from a Git
#' repository or a local file. Below are some examples of valid `packages`
#' strings:
#'
#' - Install Ruff from a specific Git tag:
#'   ```
#'   "git+https://github.com/astral-sh/ruff@v0.2.0"
#'   ```
#'
#' - Install Ruff from a specific Git commit:
#'   ```
#'   "git+https://github.com/astral-sh/ruff@1fadefa67b26508cc59cf38e6130bde2243c929d"
#'   ```
#'
#' - Install Ruff from a specific Git branch:
#'   ```
#'   "git+https://github.com/astral-sh/ruff@main"
#'   ```
#'
#' - Install MarkItDown from the `main` branch---find the package in the
#' subdirectory 'packages/markitdown':
#'   ```
#'   "markitdown@git+https://github.com/microsoft/markitdown.git@main#subdirectory=packages/markitdown"
#'   ```
#'
#' - Install MarkItDown from the local filesystem by providing an absolute path to
#' a directory containing a `pyproject.toml` or `setup.py` file:
#'   ```
#'   "markitdown@/Users/tomasz/github/microsoft/markitdown/packages/markitdown/"
#'   ```
#'
#' See more examples
#' [here](https://docs.astral.sh/uv/pip/packages/#installing-a-package) and
#' [here](https://pip.pypa.io/en/stable/cli/pip_install/#examples).
#'
#'
#' ## Clearing the Cache
#'
#' If `uv` is already installed on your machine, `reticulate` will use the
#' existing `uv` installation as-is, including its default `cache dir` location.
#' To clear the caches of a self-managed `uv` installation, send the following
#' system commands to `uv`:
#'
#' ```
#' uv cache clean
#' rm -r "$(uv python dir)"
#' rm -r "$(uv tool dir)"
#' ```
#'
#' If an existing installation of `uv` is not found, `reticulate` will
#' automatically download and store it, along with other downloaded artifacts
#' and ephemeral environments, in the `tools::R_user_dir("reticulate", "cache")`
#' directory. To clear this cache, delete the directory:
#'
#' ```r
#' # delete uv, ephemeral virtual environments, and all downloaded artifacts
#' unlink(tools::R_user_dir("reticulate", "cache"), recursive = TRUE)
#' ```
#'
#' @param packages A character vector of Python packages to be available during
#'   the session. These can be simple package names like `"jax"` or names with
#'   version constraints like `"jax[cpu]>=0.5"`. Pip style syntax for installing
#'   from local files or a git repository is also supported (see details).
#'
#' @param python_version A character vector of Python version constraints \cr
#'   (e.g., `"3.10"` or `">=3.9,<3.13"`).
#'
#' @param ... Reserved for future extensions; must be empty.
#'
#' @param action Determines how `py_require()` processes the provided
#'   requirements. Options are:
#'   - `"add"` (the default): Adds the entries to the current set of requirements.
#'   - `"remove"`: Removes _exact_ matches from the requirements list.
#'   Requests to remove nonexistent entries are ignored. For example, if
#'   `"numpy==2.2.2"` is in the list, passing `"numpy"` with `action="remove"`
#'   will not remove it.
#'   - `"set"`: Clears all existing requirements and replaces them with the
#'   provided ones. Packages and the Python version can be set independently.
#'
#' @param exclude_newer Limit package versions to those published before a
#'   specified date. This offers a lightweight alternative to freezing package
#'   versions, helping guard against Python package updates that break a
#'   workflow. Accepts strings formatted as RFC 3339 timestamps (e.g.,
#'   `"2006-12-02T02:07:43Z"`) and local dates in the same format (e.g.,
#'   `"2006-12-02"`) in your system's configured time zone. Once `exclude_newer`
#'   is set, only the `set` action can override it.
#'
#' @returns `py_require()` is primarily called for its side effect of modifying
#'   the manifest of "Python requirements" for the current R session  that
#'   reticulate maintains internally. `py_require()` usually returns `NULL`
#'   invisibly. If `py_require()` is called with no arguments, it returns the
#'   current manifest--a list with names `packages`, `python_version`, and
#'   `exclude_newer.` The list also has a class attribute, to provide a print
#'   method.
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
  ephemeral_venv_initialized <- is_epheremal_venv_initialized()
  if (missing(packages))
    packages <- NULL

  # TODO: called_from_package_onLoad <- in_onload()
  signal_and_exit <- if (called_from_package) warn_and_return else stop

  if (!is.null(python_version)) {
    python_version <- unlist(strsplit(python_version, ",", fixed = TRUE))

    if (ephemeral_venv_initialized) {

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

    if (ephemeral_venv_initialized) {

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
    if (ephemeral_venv_initialized) {
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

  if (ephemeral_venv_initialized && action == "add" && !is.null(packages)) {
    tryCatch({
      new_path <- uv_get_or_create_env(pr$packages, pr$python_version, pr$exclude_newer)
      new_config <- python_config(new_path)
      new_config$ephemeral <- TRUE
      if (new_config$libpython == .globals$py_config$libpython) {
        py_activate_virtualenv(file.path(dirname(new_path), "activate_this.py"))
        .globals$py_config <- new_config
        .globals$py_config$available <- TRUE
        # TODO: sync os.environ with R Sys.getenv()?
      } else {
        # TODO: Better error message?
        signal_and_exit(
          "New environment does not use the same Python binary\n",
          "new libpython: ", new_config$libpython, "\n",
          "old libpython: ", .globals$py_config$libpython)
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
    if(is_epheremal_venv_initialized()) {
      python_version <- paste0(
        "[No Python version specified. Defaulted to '",
        resolve_python_version() , "']"
      )
    } else {
      python_version <- paste0(
        "[No Python version specified. Will default to '",
        resolve_python_version() , "']"
      )
    }
  }

  requested_from <- as.character(lapply(x$history, function(x) x$requested_from))
  history <- x$history[requested_from != "R_GlobalEnv"]
  is_package <- as.logical(lapply(history, function(x) x$env_is_package))
  longest_pkg <- max(nchar(packages)) + 13
  if(longest_pkg < 73) {
    console_width <- 73
  } else {
    console_width <- longest_pkg
  }
  if (requireNamespace("cli", quietly = TRUE)) {
    withr::with_options(
      list("cli.width" = console_width),
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
          use_cli = TRUE,
          console_width = console_width
        ))
        cat("\n")
        if (any(is_package)) {
          cli::cli_rule("R package requests")
          py_reqs_table(history[is_package], "R package", use_cli = TRUE, console_width)
        }
        if (any(!is_package)) {
          cli::cli_rule("Environment requests")
          py_reqs_table(history[!is_package], "R package", use_cli = TRUE, console_width)
        }
      }
    )
  } else {
    pr_width <- ceiling((console_width - 21) / 2)
    cat(paste0(rep("=", pr_width), collapse = ""))
    cat(" Python requirements ")
    cat(paste0(rep("=", pr_width), collapse = ""), "\n")
    cat(
      py_reqs_format(
        packages = packages,
        python_version = python_version,
        exclude_newer = x$exclude_newer,
        console_width = console_width
        )
      )
    cat("\n")
    if (any(is_package)) {
      cat("-- R package requests ")
      cat(paste0(rep("-", console_width - 22), collapse = ""), "\n")
      py_reqs_table(history[is_package], "R package", console_width = console_width)
    }
    if (any(!is_package)) {
      cat("-- Environment requests ")
      cat(paste0(rep("-", console_width - 24), collapse = ""), "\n")
      py_reqs_table(history[!is_package], "R package", console_width = console_width)
    }
  }
  invisible()
}

# Python requirements - utils --------------------------------------------------

py_reqs_pad <- function(x = "", len, use_cli, is_title = FALSE) {

  if(nchar(x) > len) {
    x <- paste0(substr(x, 1, len-3), "...")
  }

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

py_reqs_table <- function(history, from_label, use_cli = FALSE, console_width = 73) {
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
                           use_cli = FALSE,
                           console_width = 73) {
  msg <- c(
    if (!use_cli) {
      paste0(
        "-- Current requirements ",
        paste0(rep("-", console_width - 24), collapse = ""),
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
      tbl_width <- console_width - 13
      pkg_lines <- strwrap(paste0(packages, collapse = ", "), tbl_width)
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
    if (is_epheremal_venv_initialized())
      return(as.character(py_version(TRUE)))
  }
  pr[[x]]
}

# uv ---------------------------------------------------------------------------

uv_binary <- function(bootstrap_install = TRUE) {
  min_uv_version <- numeric_version("0.6.3")
  max_uv_version <- numeric_version("0.8.0")
  is_usable_uv <- function(uv) {
    if (is.null(uv) || is.na(uv) || uv == "" || !file.exists(uv)) {
      return(FALSE)
    }
    ver <- suppressWarnings(system2(uv, "--version", stderr = TRUE, stdout = TRUE))
    if (!is.null(attr(ver, "status"))) {
      return(FALSE)
    }
    ver <- numeric_version(sub("uv ([0-9.]+).*", "\\1", ver), strict = FALSE)
    !is.na(ver) && ver >= min_uv_version && ver < max_uv_version
  }

  repeat {
    uv <- Sys.getenv("RETICULATE_UV", NA)
    if (!is.na(uv)) {
      if (uv == "managed") break else return(uv)
    }

    uv <- getOption("reticulate.uv_binary")
    if (!is.null(uv)) {
      if (uv == "managed") break else return(uv)
    }

    # on Windows, the invocation cost of `uv`` is non-negligable.
    # observed to be 0.2s for just `uv --version`
    # This is a an approach to avoid paying that cost on each invocation
    # This is mostly motivated by uv_run_tool(),
    on.exit(options(reticulate.uv_binary = uv))

    uv <- as.character(Sys.which("uv"))
    if (is_usable_uv(uv)) {
      return(uv)
    }

    uv <- path.expand("~/.local/bin/uv")
    if (is_usable_uv(uv)) {
      return(uv)
    }

    break
  }

  uv <- reticulate_cache_dir("uv", "bin", if (is_windows()) "uv.exe" else "uv")
  attr(uv, "reticulate-managed") <- TRUE
  if (is_usable_uv(uv)) {
    return(uv)
  }

  if (file.exists(uv)) {
    # exists, but version too old
    unlink(dirname(uv), recursive = TRUE)
    ## We don't do `system2(uv, "self update")` because self update is only
    ## supported on a "managed" uv installations, and uv only supports one
    ## managed installation per system. uv installs and maintains a config file
    ## for the auto updater in XDG_CONFIG_DIRS/uv/uv-receipt.json and errors if
    ## multiple uv installations attempt to modify that config file.
  }

  if (bootstrap_install) {
    # Install 'uv' in the 'r-reticulate' sub-folder inside the user's cache directory
    # https://github.com/astral-sh/uv/blob/main/docs/configuration/installer.md
    dir.create(dirname(uv), showWarnings = FALSE, recursive = TRUE)
    file_ext <- if (is_windows()) ".ps1" else ".sh"
    url <- paste0("https://astral.sh/uv/0.7.22/install", file_ext)
    install_uv <- tempfile("install-uv-", fileext = file_ext)
    download.file(url, install_uv, quiet = TRUE)
    if (!file.exists(install_uv)) {
      return(NULL)
      # stop("Unable to download Python dependencies. Please install `uv` manually.")
    }
    if (debug_uv <- Sys.getenv("_RETICULATE_DEBUG_UV_") == "1")
      system2 <- system2t

    if (is_windows()) {

      withr::with_envvar(c("UV_UNMANAGED_INSTALL" = utils::shortPathName(dirname(uv))), {
        system2("powershell", c(
          "-ExecutionPolicy", "ByPass", "-c",
          sprintf("irm %s | iex", utils::shortPathName(install_uv))),
          stdout = if (debug_uv) "" else FALSE,
          stderr = if (debug_uv) "" else FALSE
        )
      })

    } else {

      Sys.chmod(install_uv, mode = "0755")
      withr::with_envvar(c("UV_UNMANAGED_INSTALL" = dirname(uv)), {
        system2(install_uv,
                stdout = if (debug_uv) "" else FALSE,
                stderr = if (debug_uv) "" else FALSE)
      })

    }
  }

  if (file.exists(uv)) uv else NULL # print visible
}

uv_get_or_create_env <- function(packages = py_reqs_get("packages"),
                                 python_version = py_reqs_get("python_version"),
                                 exclude_newer = py_reqs_get("exclude_newer")) {

  uv <- uv_binary() %||% return() # error?

  withr::local_envvar(c(
    VIRTUAL_ENV = NA,
    if (is_positron())
      c(RUST_LOG = NA),
    if (isTRUE(attr(uv, "reticulate-managed", TRUE)))
      c(
        UV_CACHE_DIR = reticulate_cache_dir("uv", "cache"),
        UV_PYTHON_INSTALL_DIR = reticulate_cache_dir("uv", "python")
      )
  ))

  resolved_python_version <-
    resolve_python_version(constraints = python_version, uv = uv)

  if (!length(resolved_python_version)) {
    return() # error?
  }

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

  uv_output_file <- tempfile()
  on.exit(unlink(uv_output_file), add = TRUE)

  uv_args <- c(
    "run",
    "--no-project",
    # "--python-preference", "managed",
    python_version,
    exclude_newer,
    packages,
    "--",
    "python", "-c",
    # chr(119) == "w", but avoiding a string literal to minimize the need for
    # shell quoting shenanigans
    "import sys; f=open(sys.argv[-1], chr(119)); f.write(sys.executable); f.close();",
    uv_output_file
  )

  # debug print system call
  if (debug <- Sys.getenv("_RETICULATE_DEBUG_UV_") == "1")
    message(paste0(c(shQuote(uv), maybe_shQuote(uv_args)), collapse = " "))

  error_code <- suppressWarnings(system2(uv, maybe_shQuote(uv_args)))

  if (error_code) {
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

  ephemeral_python <- readLines(uv_output_file, warn = FALSE)
  if (debug)
    message("resolved ephemeral python: ", ephemeral_python)
  ephemeral_python
}

#' uv run tool
#'
#' Run a Command Line Tool distributed as a Python package. Packages are
#' automatically download and installed into a cached, ephemeral, and isolated
#' environment on the first run.
#'
#' @param tool,args A character vector of command and arguments. Arguments are
#'   not quoted for the shell, so you may need to use [`shQuote()`].
#' @param from Use the given Python package to provide the command.
#' @param with Run with the given Python packages installed. You can also
#'   specify version constraints like `"ruff>=0.3.0"`.
#' @param python_version A Python version string, or character vector of Python
#'   version constraints.
#' @param exclude_newer String. Limit package versions to those published before
#'   a specified date. This offers a lightweight alternative to freezing package
#'   versions, helping guard against Python package updates that break a
#'   workflow. Accepts strings formatted as RFC 3339 timestamps (e.g.,
#'   `"2006-12-02T02:07:43Z"`) and local dates in the same format (e.g.,
#'   `"2006-12-02"`) in your system's configured time zone.
#' @inheritDotParams base::system2 -command
#'
#' @details
#'
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
uv_run_tool <- function(tool,
                        args = character(),
                        ...,
                        from = NULL,
                        with = NULL,
                        python_version = NULL,
                        exclude_newer = NULL) {
  uv <- uv_binary()
  withr::local_envvar(c(
    VIRTUAL_ENV = NA,
    if (is_positron())
      c(RUST_LOG = NA),
    if (isTRUE(attr(uv, "reticulate-managed", TRUE)))
      c(
        UV_CACHE_DIR = reticulate_cache_dir("uv", "cache"),
        UV_PYTHON_INSTALL_DIR = reticulate_cache_dir("uv", "python")
      )
  ))

  python <- .globals$cached_uv_run_tool_python_version[[python_version %||% "default"]]
  if (is.null(python)) {
    .globals$cached_uv_run_tool_python_version[[python_version %||% "default"]] <-
      python <-
      resolve_python_version(constraints = python_version, uv = uv)
  }

  system2(uv, c(
    "tool",
    "run",
    "--isolated",
    "--python", python,
    if (length(exclude_newer)) c("--exclude-newer", exclude_newer),
    if (length(from)) c("--from", maybe_shQuote(from)),
    if (length(with)) c(rbind("--with", maybe_shQuote(with))),
    "--",
    tool,
    args
  ), ...)
}


# uv - utils -------------------------------------------------------------------


is_reticulate_managed_uv <- function(uv = uv_binary(bootstrap_install = FALSE)) {
  if (is.null(uv)) {
    # no user-installed uv - uv will be bootstrapped by reticulate
    return(TRUE)
  }

  isTRUE(attr(uv, "reticulate-managed", TRUE))
}



# return a dataframe of python options sorted by default reticulate preference
uv_python_list <- function(
  uv = uv_binary(),
  python_preference = Sys.getenv("UV_PYTHON_PREFERENCE", "only-managed")
) {
  if (isTRUE(attr(uv, "reticulate-managed", TRUE)))
    withr::local_envvar(c(
      UV_CACHE_DIR = reticulate_cache_dir("uv", "cache"),
      UV_PYTHON_INSTALL_DIR = reticulate_cache_dir("uv", "python")
    ))


  if (Sys.getenv("_RETICULATE_DEBUG_UV_") == "1")
    system2 <- system2t

  # valid values of python_preference are: only-managed, managed, system, only-system
  # https://docs.astral.sh/uv/reference/settings/#python-preference
  if (python_preference != "only-managed") {
    # uv does not find many pythons that are found by `virtualenv_starter(all=T)`,
    # including pythons installed by `install_python()`
    # To help uv find them, we temporarily place them on the PATH.
    withr::local_path(
      dirname(virtualenv_starter(all = TRUE)$path),
      action = "suffix"
    )
  }

  x <- system2(uv, c(
    "python list",
    "--all-versions",
    "--color never",
    "--output-format json",
    "--python-preference ", python_preference
    ),
    stdout = TRUE
  )

  x <- paste0(x, collapse = "")
  x <- jsonlite::parse_json(x, simplifyVector = TRUE)

  if (!length(x) &&
        missing(python_preference) &&
        is.na(Sys.getenv("UV_PYTHON_PREFERENCE", NA))) {
    return(uv_python_list(uv, "only-system"))
  }

  x <- x[is.na(x$symlink) , ]             # ignore local filesystem symlinks
  x <- x[x$variant == "default", ]        # ignore "freethreaded"
  x <- x[x$implementation == "cpython", ] # ignore "pypy"

  x$is_prerelease <- x$version != paste(x$version_parts$major,
                                        x$version_parts$minor,
                                        x$version_parts$patch,
                                        sep = ".")
  # x <- x[!x$is_prerelease, ] # ignore versions like "3.14.0a5"

  # x$path is local file path, NA if not downloaded yet.
  # x$url is populated if not downloaded yet.
  is_uv_downloadable <- !is.na(x$url)
  is_uv_downloaded <- grepl(
    "/uv/python/",
    normalizePath(as.character(x$path), winslash = "/", mustWork = FALSE),
    fixed = TRUE
  )
  x$is_uv_python <- is_uv_downloadable | is_uv_downloaded

  # order first to easily resolve the latest preferred patch for each minor version
  x <- x[order(
    !x$is_prerelease,
    x$is_uv_python,
    x$version_parts$major,
    x$version_parts$minor,
    x$version_parts$patch,
    decreasing = TRUE
  ), ]

  # Order so the latest patch level for each minor version appears first,
  # prioritizing two versions behind the latest minor release.
  # Sort by the distance of the minor version from the preferred minor version,
  # breaking ties in favor of older minor versions.
  latest_minor <- max(x$version_parts$minor[!x$is_prerelease])
  preferred_minor <- latest_minor - 2L
  x$is_latest_patch <- !duplicated(x$version_parts[c("major", "minor")])

  x <- x[order(
    !x$is_prerelease,
    x$is_uv_python,
    x$is_latest_patch,
    -abs(x$version_parts$minor - preferred_minor) +
      (-0.5 * (x$version_parts$minor > preferred_minor)),
    x$version_parts$major == 3L,
    x$version_parts$minor,
    x$version_parts$patch,
    decreasing = TRUE
  ), ]

  x
}

uvx_binary <- function(...) {
  uv <- uv_binary(...)
  if(is.null(uv)) {
    return()
  }
  uvx <- file.path(dirname(uv), if (is_windows()) "uvx.exe" else "uvx")
  if (file.exists(uvx)) uvx else NULL # print visible
}

uv_exec <- function(args, ...) {
  uv <- uv_binary()
  withr::local_envvar(c(
    VIRTUAL_ENV = NA,
    if (is_positron())
      c(RUST_LOG = NA),
    if (isTRUE(attr(uv, "reticulate-managed", TRUE)))
      c(
        UV_CACHE_DIR = reticulate_cache_dir("uv", "cache"),
        UV_PYTHON_INSTALL_DIR = reticulate_cache_dir("uv", "python")
      )
  ))

  system2(uv, args, ...)
}

resolve_python_version <- function(constraints = NULL, uv = uv_binary()) {
  constraints <- as.character(constraints %||% "")
  constraints <- trimws(unlist(strsplit(constraints, ",", fixed = TRUE)))
  constraints <- constraints[nzchar(constraints)]

  # We perform custom constraint resolution to prefer slightly older Python releases.
  # uv tends to select the latest version, which often lack package support
  # See: https://devguide.python.org/versions/

  # Get latest patch for each minor version
  # E.g., candidates might be:
  #  c("3.13.1", "3.12.8", "3.11.11", "3.10.16", "3.9.21", "3.8.20" , ...)
  all_candidates <- candidates <- uv_python_list(uv)$version

  if (length(constraints) == 0L) {
    return(as.character(candidates[1L])) # default
  }

  # reflect a direct version specification like "3.14.0a3"
  if (length(constraints) == 1L && constraints %in% candidates) {
    return(constraints)
  }

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
      'Available Python versions found: ', paste0(all_candidates, collapse = ", "), "\n",
      'Hint: Call `py_require(python_version = <string>, action = "set")` to replace constraints.'
    )
    stop(msg)
  }

  as.character(candidates[1L])
}


uv_diff_exclude_newer <- function(from = -3L, to = Sys.Date(),
                                  packages = py_reqs_get("packages"),
                                  python_version = py_reqs_get("python_version"),
                                  show = TRUE) {
  uv <- uv_binary()
  if(rlang::is_bare_numeric(from))
    from <- to + from
  from <- format(from)
  to <- format(to)
  manifest <- lapply(list(from = from, to = to), function(exclude_newer) {
    python <- uv_get_or_create_env(packages, python_version, exclude_newer)
    manifest <- jsonlite::parse_json(system2(
      uv,
      c("pip list --quiet --format json --python", shQuote(python)),
      stdout = TRUE
    ), simplifyVector = TRUE)
    attr(manifest, "python") <- python
    attr(manifest, "exclude_newer") <- exclude_newer
    manifest
  })

  if (show) {
    rlang::check_installed("diffobj")
    print(
      asNamespace("diffobj")$diffPrint(
        manifest$from,
        manifest$to,
        tar.banner = sprintf("from: %s", from),
        cur.banner = sprintf("to: %s", to)
      )
    )
  }
  manifest
}
