# uv ---------------------------------------------------------------------------

download_uv_installer <- function() {
  file_ext <- if (is_windows()) ".ps1" else ".sh"
  installer <- tempfile("install-uv-", fileext = file_ext)
  downloaded <- FALSE
  on.exit(if (!downloaded) unlink(installer), add = TRUE)

  status <- download.file(
    paste0("https://astral.sh/uv/install", file_ext),
    installer,
    quiet = TRUE
  )
  if (!identical(status, 0L) || !file.exists(installer))
    return()

  downloaded <- TRUE
  installer
}

uv_binary <- function(bootstrap_install = TRUE) {
  # Fast paths: use a configured or cached binary without probing it.
  configured_uv <- Sys.getenv("RETICULATE_UV", unset = NA)
  if (!is.na(configured_uv) && !identical(configured_uv, "managed"))
    return(configured_uv)

  cached_uv <- getOption("reticulate.uv_binary")
  if (is.na(configured_uv) &&
      length(cached_uv) &&
      !identical(cached_uv, "managed"))
    return(cached_uv)

  if (identical(configured_uv, "managed") &&
      isTRUE(attr(cached_uv, "reticulate-managed", exact = TRUE)))
    return(cached_uv)

  force_managed <- identical(configured_uv, "managed") ||
    identical(cached_uv, "managed")

  # Slow path: discover and validate a user-installed uv.
  if (!force_managed) {
    candidates <- c(as.character(Sys.which("uv")), path.expand("~/.local/bin/uv"))
    for (uv in candidates) {
      if (uv_is_usable(uv)) {
        options(reticulate.uv_binary = uv)
        return(uv)
      }
    }
  }

  # Managed path: validate the cached binary or install it.
  uv <- reticulate_cache_dir(
    "uv", "bin", if (is_windows()) "uv.exe" else "uv"
  )
  attr(uv, "reticulate-managed") <- TRUE
  installer <- NULL

  if (bootstrap_install) {
    installer <- maybe_clear_reticulate_uv_cache(uv)
    if (!is.null(installer))
      on.exit(unlink(installer), add = TRUE)
  }

  if (!uv_is_usable(uv)) {
    if (file.exists(uv))
      unlink(dirname(dirname(uv)), recursive = TRUE, force = TRUE)

    if (!bootstrap_install)
      return()

    if (is.null(installer)) {
      message("Downloading uv...", appendLF = FALSE)
      installer <- download_uv_installer()
      if (is.null(installer))
        return()
      on.exit(unlink(installer), add = TRUE)
      message("Done!")
    }

    uv_install_managed(uv, installer)
    if (!uv_is_usable(uv)) {
      stop(
        "uv bootstrap failed: installed uv binary is not usable.",
        call. = FALSE
      )
    }
  }

  # Keep RETICULATE_UV as configuration and cache the attributed result in R.
  if (bootstrap_install)
    options(reticulate.uv_binary = uv)

  uv
}


uv_is_usable <- function(uv) {
  if (!length(uv) || is.na(uv) || !nzchar(uv) || !file.exists(uv))
    return(FALSE)

  version <- suppressWarnings(
    system2(uv, "--version", stderr = TRUE, stdout = TRUE)
  )
  if (!is.null(attr(version, "status")))
    return(FALSE)

  version <- numeric_version(
    sub("uv ([0-9.]+).*", "\\1", version),
    strict = FALSE
  )
  !is.na(version) && version >= "0.6.3"
}


uv_install_managed <- function(uv, installer) {
  # The installer places uv in the bin directory below UV_UNMANAGED_INSTALL.
  # Start from an empty directory in case a previous installation was interrupted.
  unlink(dirname(dirname(uv)), recursive = TRUE, force = TRUE)
  dir.create(dirname(uv), showWarnings = FALSE, recursive = TRUE)

  debug <- Sys.getenv("_RETICULATE_DEBUG_UV_") == "1"
  if (debug)
    system2 <- system2t

  stdout <- tempfile("install-uv-stdout-")
  stderr <- tempfile("install-uv-stderr-")
  on.exit(unlink(c(stdout, stderr)), add = TRUE)

  if (is_windows()) {
    withr::with_envvar(c(
      UV_UNMANAGED_INSTALL = utils::shortPathName(dirname(uv)),
      PSModulePath = NA
    ), {
      status <- system2(
        "powershell.exe",
        c(
          "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
          "-File", utils::shortPathName(installer)
        ),
        stdout = stdout,
        stderr = stderr
      )
    })
  } else {
    Sys.chmod(installer, mode = "0755")
    withr::with_envvar(c(UV_UNMANAGED_INSTALL = dirname(uv)), {
      status <- system2(installer, stdout = stdout, stderr = stderr)
    })
  }

  stdout_lines <- if (file.exists(stdout)) readLines(stdout, warn = FALSE)
  stderr_lines <- if (file.exists(stderr)) readLines(stderr, warn = FALSE)
  if (debug) {
    writeLines(stdout_lines)
    writeLines(stderr_lines, con = base::stderr())
  }

  if (identical(status, 0L) && file.exists(uv))
    return(invisible())

  details <- c(
    if (length(stdout_lines)) c("stdout:", paste0("  ", stdout_lines)),
    if (length(stderr_lines)) c("stderr:", paste0("  ", stderr_lines))
  )
  msg <- if (!identical(status, 0L)) {
    sprintf("uv bootstrap failed with exit status %s.", status)
  } else {
    sprintf(
      "uv bootstrap failed: installer completed without creating %s.",
      shQuote(uv)
    )
  }
  stop(paste(c(msg, details), collapse = "\n"), call. = FALSE)
}

uv_get_or_create_env <- function(packages = py_reqs_get()$packages,
                                 python_version = py_reqs_python_version(),
                                 exclude_newer = py_reqs_get()$exclude_newer) {

  uv <- uv_binary() %||% return()

  resolved_python_version <-
    resolve_python_version(constraints = python_version, uv = uv)

  if (!length(resolved_python_version))
    return()

  call_args <- list(
    packages = packages,
    python_version = python_version %||%
      paste(resolved_python_version, "(reticulate default)"),
    exclude_newer = exclude_newer
  )

  if (length(packages))
    packages <- as.vector(rbind("--with", packages))

  python_version <- c("--python", resolved_python_version)

  if (!is.null(exclude_newer))
    exclude_newer <- c("--exclude-newer", exclude_newer)

  uv_output_file <- tempfile()
  on.exit(unlink(uv_output_file), add = TRUE)

  uv_args <- c(
    "tool", "run",
    "--isolated",
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

  error_code <- suppressWarnings(uv_exec(maybe_shQuote(uv_args), uv = uv))

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

    if (any(call_args$packages %in% builtin_module_names)) {
      invalid <- unique(c(
        "sys", "os", intersect(call_args$packages, builtin_module_names)
      ))
      writeLines(con = stderr(), c(
        "Hint: `py_require()` expects Python package names rather than Python module names.",
        sprintf(
          "Modules provided by the Python standard library such as %s should not be passed to `py_require()`.",
          pc_and("`", invalid, "`")
        ),
        strrep("-", 73L)
      ))
    }

    stop("Call `py_require()` to remove or replace conflicting requirements.")
  }

  cached_python <- readLines(uv_output_file, warn = FALSE)
  if (Sys.getenv("_RETICULATE_DEBUG_UV_") == "1")
    message("resolved ephemeral python: ", cached_python)
  cached_python
}


py_reqs_python_version <- function() {
  if (is_ephemeral_venv_initialized())
    return(as.character(py_version(patch = TRUE)))

  py_reqs_get()$python_version
}


# uv_get_or_create_env(packages = NULL) |>
#   system2("-", stdout = TRUE, input = '
# import pkgutil
#
# modules = [
#     module.name
#     for module in pkgutil.iter_modules()
#     if not module.name.startswith("_")
# ]
#
# print("c", tuple(sorted(modules)), sep = "")
# ') |>
#   clipr::write_clip()

builtin_module_names <- c('abc', 'aifc', 'antigravity', 'argparse', 'ast', 'asynchat', 'asyncio', 'asyncore', 'base64', 'bdb', 'bisect', 'bz2', 'cProfile', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmd', 'code', 'codecs', 'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg', 'crypt', 'csv', 'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings', 'ensurepip', 'enum', 'filecmp', 'fileinput', 'fnmatch', 'fractions', 'ftplib', 'functools', 'genericpath', 'getopt', 'getpass', 'gettext', 'glob', 'graphlib', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'mimetypes', 'modulefinder', 'multiprocessing', 'netrc', 'nntplib', 'ntpath', 'nturl2path', 'numbers', 'opcode', 'operator', 'optparse', 'os', 'pathlib', 'pdb', 'pickle', 'pickletools', 'pip', 'pipes', 'pkg_resources', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'py_compile', 'pyclbr', 'pydoc', 'pydoc_data', 'queue', 'quopri', 'random', 're', 'reprlib', 'rlcompleter', 'runpy', 'sched', 'secrets', 'selectors', 'setuptools', 'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'sqlite3', 'sre_compile', 'sre_constants', 'sre_parse', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess', 'sunau', 'symtable', 'sysconfig', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'textwrap', 'this', 'threading', 'timeit', 'tkinter', 'tm', 'token', 'tokenize', 'tomllib', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing', 'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zoneinfo')

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

  key <- python_version %||% "default"
  python <- .globals$cached_uv_run_tool_python_version[[key]]
  if (is.null(python)) {
    .globals$cached_uv_run_tool_python_version[[key]] <-
      python <-
      resolve_python_version(constraints = python_version, uv = uv)
  }

  uv_exec(c(
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
  ), ..., uv = uv)
}


# uv - utils -------------------------------------------------------------------


# return a dataframe of python options sorted by default reticulate preference
uv_python_list <- function(
  uv = uv_binary(),
  python_preference = Sys.getenv("UV_PYTHON_PREFERENCE", "only-managed")
) {
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

  x <- uv_exec(c(
    "python list",
    "--all-versions",
    "--color never",
    "--output-format json",
    "--python-preference ", python_preference
    ),
    stdout = TRUE,
    uv = uv
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

uv_exec <- function(args, ..., uv = uv_binary()) {
  withr::local_envvar(c(
    VIRTUAL_ENV = NA,
    if (is_positron())
      c(RUST_LOG = NA),
    if (isTRUE(attr(uv, "reticulate-managed", exact = TRUE)))
      c(
        UV_CACHE_DIR = reticulate_cache_dir("uv", "cache"),
        UV_PYTHON_INSTALL_DIR = reticulate_cache_dir("uv", "python")
      )
  ))

  if (Sys.getenv("_RETICULATE_DEBUG_UV_") == "1")
    system2 <- system2t

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
      'Hint: Call `py_require(python_version = <string>, action = "set")` to replace constraints.\n',
      'Available Python versions found: ', paste0(all_candidates, collapse = ", "), "\n"
    )
    stop(msg)
  }

  as.character(candidates[1L])
}


maybe_clear_reticulate_uv_cache <- function(uv) {
  if (!file.exists(uv))
    return()

  max_age <- Sys.getenv("RETICULATE_MAX_CACHE_AGE_DAYS", unset = NA)
  if (!is.na(max_age)) {
    max_age <- suppressWarnings(as.numeric(max_age))
    max_age <- as.difftime(max_age, units = "days")
  } else {
    max_age <- getOption(
      "reticulate.max_cache_age",
      as.difftime(120, units = "days")
    )
  }
  if (is.na(max_age))
    return()
  if (!inherits(max_age, "difftime"))
    return()

  uv_ctime <- file.info(uv, extra_cols = FALSE)$ctime
  actual_age <- difftime(Sys.time(), uv_ctime, units = units(max_age))
  if (actual_age <= max_age)
    return()

  if (Sys.getenv("UV_OFFLINE") == "1")
    return()

  installer <- suppressWarnings(try(
    download_uv_installer(),
    silent = TRUE
  ))
  if (inherits(installer, "try-error") || is.null(installer)) {
    message(
      "Retaining reticulate's uv cache because access to the uv ",
      "installer could not be verified."
    )
    return()
  }

  cache_dir <- dirname(dirname(uv))
  # best-effort; avoid surfacing errors
  message("Clearing reticulate's uv cache...", appendLF = FALSE)
  tryCatch(
    {
      # Delete the uv binary first, so if the unlink(cache_dir) call is interrupted,
      # the cache is still invalidated and we trigger a fresh bootstrap install on next run.
      # The delete command is re-run before bootstrapping to double-check/confirm
      # the cache_dir is empty.
      unlink(uv, force = TRUE)
      unlink(cache_dir, recursive = TRUE, force = TRUE)
    },
    error = warning
  )
  message("Done!")
  installer
}
