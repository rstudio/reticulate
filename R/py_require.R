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

  pr <- py_reqs_get()

  if (missing(packages) && missing(python_version) && missing(exclude_newer)) {
    return(pr)
  }

  action <- match.arg(action)
  called_from_package <- isNamespace(topenv(parent.frame()))
  uv_initialized <- is_python_initialized() && is_ephemeral_reticulate_uv_env(py_exe())

  if (!is.null(python_version)) {
    python_version <- unlist(strsplit(python_version, ",", fixed = TRUE))

    if (uv_initialized) {

      current_py_version <- py_version(patch = TRUE)
      for (check in as_version_constraint_checkers(python_version)) {
        if (!isTRUE(check(current_py_version))) {
          signal <- if(called_from_package) warning else stop
          signal(
            "Python version requirements cannot be ",
            "changed after Python has been initialized.\n",
            "- Python version request: '", python_version, "'",
            if (called_from_package) paste0(" (from package:", parent.pkg(), ")"),
            "\n",
            "- Python version initialized: '", as.character(current_py_version), "'"
          )
          break
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
    if (action != "set" && !is.null(py_reqs_get("exclude_newer"))) {
      stop(
        "`exclude_newer` is already set to '",
        py_reqs_get("exclude_newer"),
        "', use `action = 'set'` to override"
      )
    }
  }


  pr$packages <- py_reqs_action(action, packages, py_reqs_get("packages"))
  pr$exclude_newer <- pr$exclude_newer %||% exclude_newer
  pr$history <- c(pr$history, list(list(
    requested_from = environmentName(topenv(parent.frame())),
    env_is_package = called_from_package,
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
      # TODO: sync os.environ with R Sys.setvar?
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

  uv <- path.expand(file.path(
    rappdirs::user_cache_dir("r-reticulate", NULL),
    "bin", if (is_windows()) "uv.exe" else "uv"
  ))
  if (file.exists(uv)) {
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

  # capture args; maybe used in error message later
  call_args <- list(
    packages = packages,
    python_version = python_version %||%
      paste(resolve_python_version(), "(reticulate default)"),
    exclude_newer = exclude_newer
  )

  if (length(packages))
    packages <- as.vector(rbind("--with", packages))

  python_version <- c("--python", resolve_python_version(constraints = python_version))

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

  # "tool",
  # "--no-config",
  # "--isolated",
  # "--upgrade",

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

  ## debug print system call:
  # message(paste0(c(uv, maybe_shQuote(uv_args)), collapse = " "))

  env_python <- suppressWarnings(system2(uv, maybe_shQuote(uv_args), stdout = TRUE))
  exit_status <- attr(env_python, "status", TRUE) %||% 0L

  if (exit_status != 0L) {
    msg <- do.call(py_reqs_format, call_args)
    writeLines(c(msg, strrep("-", 73L)), con = stderr())
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


uv_python_list <- function() {
  x <- system2(uv_binary(), c("python list",
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

resolve_python_version <- function(constraints = NULL) {
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
  candidates <- uv_python_list()
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
