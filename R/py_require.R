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
#' directory. Set `R_USER_CACHE_DIR` to configure this location; see
#' `tools::R_user_dir()` for details. To clear this cache manually, delete the
#' directory:
#'
#' ```r
#' # delete uv, ephemeral virtual environments, and all downloaded artifacts
#' unlink(tools::R_user_dir("reticulate", "cache"), recursive = TRUE)
#' ```
#'
#' Reticulate also clears its managed cache automatically on an interval,
#' defaulting to every 120 days. Set `RETICULATE_MAX_CACHE_AGE_DAYS` to a
#' possibly fractional number of days in `.Renviron`:
#'
#' ```
#' RETICULATE_MAX_CACHE_AGE_DAYS=30
#' ```
#'
#' The environment variable takes precedence over the
#' `reticulate.max_cache_age` R option. Configure the option in `.Rprofile`
#' with:
#'
#' ```r
#' options(reticulate.max_cache_age = as.difftime(30, units = "days"))
#' ```
#'
#' Before cleanup, reticulate downloads the `uv` installer and reuses it if a
#' replacement is needed. If the download fails, reticulate retains the expired
#' cache and defers cleanup.
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
#'   current manifest--a list with names `python_version`, `packages`,
#'   `exclude_newer`, and `history`. `history` is an append-only record of
#'   successful requests, retained to help diagnose where requirements came
#'   from. It is not used to resolve the manifest. The list also has a class
#'   attribute, to provide a print method.
#'
#' @export
py_require <- function(packages = NULL,
                       python_version = NULL,
                       ...,
                       exclude_newer = NULL,
                       action = c("add", "remove", "set")) {
  if (length(list(...)))
    stop("... must be empty")

  if (missing(packages) && missing(python_version) && missing(exclude_newer)) {
    return(py_reqs_get())
  }

  caller <- topenv(parent.frame())
  called_from_package <- isNamespace(caller)
  request <- py_reqs_request(
    packages = if (missing(packages)) NULL else packages,
    python_version = if (missing(python_version)) NULL else python_version,
    exclude_newer = if (missing(exclude_newer)) NULL else exclude_newer,
    exclude_newer_supplied =
      !missing(exclude_newer) && !is.null(exclude_newer),
    action = match.arg(action),
    source = environmentName(caller),
    source_is_package = called_from_package
  )

  if (request$exclude_newer_supplied && called_from_package)
    stop("`exclude_newer` cannot be set inside a package")

  transition <- tryCatch(
    py_reqs_transition(
      current = py_reqs_get(),
      request = request,
      initialized = is_ephemeral_venv_initialized()
    ),
    error = identity
  )
  if (inherits(transition, "error")) {
    if (called_from_package) {
      warning(conditionMessage(transition))
      return(invisible())
    }
    stop(conditionMessage(transition))
  }

  if (!is.null(transition$config))
    .globals$py_config <- transition$config

  .globals$python_requirements <- transition$manifest
  invisible()
}


py_reqs_request <- function(packages,
                            python_version,
                            exclude_newer,
                            exclude_newer_supplied,
                            action,
                            source,
                            source_is_package) {
  if (!is.null(python_version)) {
    python_version <- trimws(unlist(
      strsplit(python_version, ",", fixed = TRUE),
      use.names = FALSE
    ))
  }

  if (exclude_newer_supplied) {
    if (length(exclude_newer) != 1L)
      stop("`exclude_newer` must be a single value")
    if (is.na(exclude_newer) || identical(exclude_newer, ""))
      exclude_newer <- NULL
  }

  list(
    packages = packages,
    python_version = python_version,
    exclude_newer = exclude_newer,
    exclude_newer_supplied = exclude_newer_supplied,
    action = action,
    source = source,
    source_is_package = source_is_package
  )
}


py_reqs_transition <- function(current, request, initialized) {
  plan <- py_reqs_plan(current, request, initialized)
  config <- if (plan$activate)
    py_reqs_activate(plan$manifest)
  manifest <- py_reqs_record_history(plan$manifest, request)

  list(manifest = manifest, config = config)
}


py_reqs_plan <- function(current, request, initialized) {
  if (!initialized) {
    return(list(
      manifest = py_reqs_apply(current, request),
      activate = FALSE
    ))
  }

  candidate <- current
  activate <- FALSE

  if (!is.null(request$python_version)) {
    current_version <- py_version(patch = TRUE)
    for (check in as_version_constraint_checkers(request$python_version)) {
      if (!isTRUE(check(current_version))) {
        stop(paste0(
          "Python version requirements cannot be changed after Python has ",
          "been initialized.\n",
          "* Python version request: '",
          paste(request$python_version, collapse = ","), "'",
          if (request$source_is_package)
            paste0(" (from package:", request$source, ")"),
          "\n* Python version initialized: '", current_version, "'"
        ))
      }
    }
  }

  if (request$exclude_newer_supplied &&
      !identical(request$exclude_newer, current$exclude_newer)) {
    stop("`exclude_newer` cannot be changed after Python has initialized.")
  }

  if (!is.null(request$packages)) {
    switch(request$action,
      add = {
        added <- setdiff(request$packages, current$packages)
        if (length(added)) {
          added_names <- py_requirement_name(added)
          current_names <- py_requirement_name(current$packages)
          conflicts <- added_names %in% current_names
          if (any(conflicts)) {
            new <- paste0("`", sort(added[conflicts]), "`", collapse = ", ")
            old <- current$packages[current_names %in% added_names[conflicts]]
            old <- paste0("`", sort(old), "`", collapse = ", ")
            stop(paste(
              "After Python has initialized, only `action = 'add'` with new packages is supported.",
              "You tried to add", new, "but requirements contain", old, "already."
            ))
          }
          candidate$packages <- unique(c(added, current$packages))
          activate <- TRUE
        }
      },
      remove = {
        if (any(request$packages %in% current$packages))
          stop("After Python has initialized, only `action = 'add'` is supported.")
      },
      set = {
        if (!setequal(request$packages, current$packages))
          stop("After Python has initialized, only `action = 'add'` is supported.")
      }
    )
  }

  list(manifest = candidate, activate = activate)
}


py_reqs_apply <- function(current, request) {
  candidate <- current

  if (!is.null(request$packages)) {
    candidate$packages <- py_reqs_action(
      request$action,
      request$packages,
      current$packages
    )
  }

  if (!is.null(request$python_version)) {
    candidate$python_version <- py_reqs_action(
      request$action,
      request$python_version,
      current$python_version
    )
  }

  if (request$exclude_newer_supplied) {
    exclude_newer <- switch(request$action,
      add = {
        if (!is.null(current$exclude_newer)) {
          stop(
            "`exclude_newer` is already set to '", current$exclude_newer,
            "', use `action = 'set'` to override"
          )
        }
        request$exclude_newer
      },
      remove = {
        if (is.null(request$exclude_newer) ||
            identical(request$exclude_newer, current$exclude_newer))
          NULL
        else
          current$exclude_newer
      },
      set = request$exclude_newer
    )
    candidate["exclude_newer"] <- list(exclude_newer)
  }

  candidate
}


py_requirement_name <- function(requirement) {
  requirement <- trimws(requirement, which = "left")
  match <- regexpr("^[[:alnum:]][[:alnum:]_.-]*", requirement)
  match_length <- attr(match, "match.length")
  named <- match == 1L
  remainder <- requirement
  remainder[named] <- substring(
    requirement[named],
    match_length[named] + 1L
  )
  named <- named & grepl("^$|^[[:space:]\\[(@<>=!~;]", remainder)

  name <- requirement
  name[named] <- substring(requirement[named], 1L, match_length[named])
  name[named] <- tolower(gsub("[-_.]+", "-", name[named]))
  name
}


py_reqs_activate <- function(manifest) {
  new_path <- uv_get_or_create_env(
    packages = manifest$packages,
    python_version = as.character(py_version(patch = TRUE)),
    exclude_newer = manifest$exclude_newer
  )
  new_config <- python_config(new_path)
  new_config$ephemeral <- TRUE

  if (!identical(new_config$libpython, .globals$py_config$libpython)) {
    stop(
      "New environment does not use the same Python binary\n",
      "new libpython: ", new_config$libpython, "\n",
      "old libpython: ", .globals$py_config$libpython
    )
  }

  py_activate_virtualenv(file.path(dirname(new_path), "activate_this.py"))
  new_config$available <- TRUE
  new_config
}


#' @export
print.python_requirements <- function(x, ...) {
  writeLines(format(x, ...))
  invisible()
}

#' @export
format.python_requirements <- function(x, ..., width = 73L) {
  field <- function(label, value, indent = 2L, empty = NULL) {
    if (!length(value))
      value <- empty
    if (!length(value))
      return(character())
    prefix <- paste0(label, ": ")
    strwrap(
      paste0(prefix, paste(value, collapse = ", ")),
      width = width,
      indent = indent,
      exdent = indent + nchar(prefix)
    )
  }

  python_version <- x$python_version
  if (!length(python_version)) {
    default <- resolve_python_version()
    default_message <- if (is_ephemeral_venv_initialized())
      "Defaulted"
    else
      "Will default"
    python_version <- sprintf(
      "[No Python version specified. %s to '%s']",
      default_message,
      default
    )
  }

  out <- c(
    "Python requirements:",
    field("Python", python_version),
    field("Packages", x$packages, empty = "[No packages specified]"),
    if (length(x$exclude_newer))
      field("Exclude", paste("Anything newer than", x$exclude_newer))
  )

  history <- Filter(
    function(event) !identical(event$requested_from, "R_GlobalEnv"),
    x$history
  )
  if (!length(history))
    return(out)

  out <- c(out, "Python requirement requests (in order):")
  for (i in seq_along(history)) {
    event <- history[[i]]
    source <- if (event$env_is_package) {
      paste("R package", event$requested_from)
    } else if (nzchar(event$requested_from)) {
      paste("R environment", event$requested_from)
    } else {
      "R environment"
    }
    out <- c(
      out,
      sprintf("  %d. %s:", i, source),
      field("Action", event$action, indent = 4L),
      field("Packages", event$packages, indent = 4L),
      field("Python", event$python_version, indent = 4L),
      if (isTRUE(event$exclude_newer_supplied))
        field(
          "Exclude",
          event$exclude_newer,
          indent = 4L,
          empty = "[No cutoff]"
        )
    )
  }
  out
}


# Python requirements - utils --------------------------------------------------

py_reqs_action <- function(action, x, current = NULL) {
  switch(action,
    add = unique(c(current, x)),
    remove = setdiff(current, x),
    set = x
  )
}


py_reqs_record_history <- function(manifest, request) {
  event <- list(
    requested_from = request$source,
    env_is_package = request$source_is_package,
    packages = request$packages,
    python_version = request$python_version,
    exclude_newer = request$exclude_newer,
    exclude_newer_supplied = request$exclude_newer_supplied,
    action = request$action
  )
  manifest$history <- c(manifest$history, list(event))
  manifest
}


py_reqs_format <- function(packages = NULL,
                           python_version = NULL,
                           exclude_newer = NULL,
                           console_width = 73L) {
  package_lines <- if (length(packages)) {
    packages <- strwrap(
      paste(packages, collapse = ", "),
      width = console_width - 13L
    )
    paste0(
      c(" Packages: ", rep("           ", length(packages) - 1L)),
      packages
    )
  }

  paste(c(
    paste0("-- Current requirements ", strrep("-", console_width - 24L)),
    if (length(python_version))
      paste0(" Python:   ", paste(python_version, collapse = ", ")),
    package_lines,
    if (length(exclude_newer))
      paste0(" Exclude:  Anything newer than ", exclude_newer)
  ), collapse = "\n")
}


py_reqs_get <- function() {
  manifest <- .globals$python_requirements
  if (!is.null(manifest))
    return(manifest)

  packages <- c("numpy", if (is_positron()) "ipykernel")
  manifest <- structure(
    list(
      python_version = NULL,
      packages = packages,
      exclude_newer = NULL,
      history = list(list(
        requested_from = "reticulate",
        env_is_package = TRUE,
        action = "add",
        packages = packages,
        exclude_newer_supplied = FALSE
      ))
    ),
    class = "python_requirements"
  )
  .globals$python_requirements <- manifest
  manifest
}
