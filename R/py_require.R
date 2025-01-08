#' @export
py_require <- function(packages = NULL,
                       python_version = NULL,
                       exclude_newer = NULL,
                       action = c("add", "omit", "set"),
                       silent = TRUE) {
  action <- match.arg(action)

  current_exclude <- get_python_reqs("exclude_newer")

  if (!is.null(exclude_newer) && action != "set" && !is.null(current_exclude)) {
    stop(
      "`exclude_newer` is already set to '",
      current_exclude,
      "', use `action` 'set' to override"
    )
  }

  # Priming with `numpy` if empty
  if (is.null(get_python_reqs("packages"))) {
    set_python_reqs(
      packages = "numpy",
      history = list(list(
        requested_from = "reticulate",
        requested_is_package = TRUE,
        packages = "numpy",
        action = "add",
        python_version = NULL
      ))
    )
  }

  if (missing(packages) && missing(python_version) && missing(exclude_newer)) {
    return(get_python_reqs())
  }

  req_packages <- NULL
  if (!is.null(packages)) {
    req_packages <- get_python_reqs("packages")
    packages <- switch(action,
      add = unique(c(req_packages, packages)),
      omit = {
        for (pkg in packages) {
          matches <- pkg == req_packages
          req_packages <- req_packages[!matches]
        }
        req_packages
      },
      set = packages
    )
  }

  if (!is.null(python_version)) {
    req_python_versions <- get_python_reqs("python_version")
    python_version <- switch(action,
      add = unique(c(python_version, req_python_versions)),
      omit = req_python_versions[req_python_versions != python_version],
      set = python_version
    )
  }

  top_env <- topenv(parent.frame())

  new_history <- c(
    get_python_reqs("history"),
    list(list(
      requested_from = environmentName(top_env),
      requested_is_package = isNamespace(top_env),
      packages = packages,
      action = action,
      python_version = python_version,
      exclude_newer = exclude_newer
    ))
  )

  set_python_reqs(
    packages = packages,
    python_versions = python_version,
    exclude_newer = exclude_newer,
    history = new_history
  )

  invisible()
}

# Print ------------------------------------------------------------------------

#' @export
print.python_requirements <- function(x, ...) {
  packages <- x$packages
  if (is.null(packages)) {
    packages <- "[No packages added yet]"
  } else {
    packages <- paste0(packages, collapse = ", ")
  }
  python_versions <- x$python_versions
  if (is.null(python_versions)) {
    python_versions <- "[No version of Python selected yet]"
  } else {
    python_versions <- paste0(python_versions, collapse = ", ")
  }
  cat("------------------ Python requirements ----------------------\n")
  cat("Current requirements ----------------------------------------\n")
  cat(" Packages:", packages, "\n")
  cat(" Python:  ", python_versions, "\n")
  if (!is.null(x$exclude_newer)) {
    cat(" Exclude: After", x$exclude_newer, "\n")
  }
  cat("Non-user requirements ---------------------------------------\n")
  reqs <- data.frame()
  for (item in x$history) {
    if (item$requested_from != "R_GlobalEnv") {
      if (item$requested_is_package) {
        req_label <- paste0("package:", item$requested_from)
      } else {
        req_label <- item$requested_from
      }
      row_x <- data.frame(
        requestor = paste(req_label, "|"),
        packages = paste(paste0(item$packages, collapse = ", "), "|"),
        python = paste(paste0(item$python, collapse = ", "), "|")
      )
      reqs <- rbind(reqs, row_x)
    }
  }
  if (!is.null(reqs)) {
    print(reqs)
  }
  return(invisible())
}

# Get/set requirements ---------------------------------------------------------

get_python_reqs <- function(
    x = c("all", "python_versions", "packages", "exclude_newer", "history")) {
  pr <- .globals$python_requirements
  x <- match.arg(x)
  switch(x,
    all = pr,
    python_versions = pr$python_versions,
    packages = pr$packages,
    exclude_newer = pr$exclude_newer,
    history = pr$history
  )
}

set_python_reqs <- function(
    python_versions = NULL,
    packages = NULL,
    exclude_newer = NULL,
    history = NULL) {
  pr <- get_python_reqs("all")
  pr$python_versions <- python_versions %||% pr$python_versions
  pr$packages <- packages %||% pr$packages
  pr$exclude_newer <- exclude_newer %||% pr$exclude_newer
  pr$history <- history %||% pr$history
  .globals$python_requirements <- pr
  get_python_reqs("all")
}

.globals$python_requirements <- structure(
  .Data = list(
    python_versions = c(),
    packages = c(),
    exclude_newer = NULL,
    history = list()
  ),
  class = "python_requirements"
)

# uv ---------------------------------------------------------------------------

uv_binary <- function() {
  uv <- Sys.getenv("RETICULATE_UV", NA)
  if (!is.na(uv)) {
    return(uv)
  }

  uv <- getOption("reticulate.uv_binary")
  if (!is.null(uv)) {
    return(uv)
  }

  uv <- as.character(Sys.which("uv"))
  if (uv != "") {
    return(uv)
  }

  uv <- path.expand("~/.local/bin/uv")
  if (file.exists(uv)) {
    return(uv)
  }

  uv <- file.path(rappdirs::user_cache_dir("r-reticulate", NULL), "bin", "uv")
  if (file.exists(uv)) {
    return(uv)
  }

  if (is_windows()) {

  } else if (is_macos() || is_linux()) {
    install_uv.sh <- tempfile("install-uv-", fileext = ".sh")
    download.file("https://astral.sh/uv/install.sh", install_uv.sh, quiet = TRUE)
    Sys.chmod(install_uv.sh, mode = "0755")

    dir.create(dirname(uv), showWarnings = FALSE)
    # https://github.com/astral-sh/uv/blob/main/docs/configuration/installer.md
    system2(install_uv.sh, c("--quiet"), env = c(
      "INSTALLER_NO_MODIFY_PATH=1", paste0("UV_INSTALL_DIR=", maybe_shQuote(dirname(uv)))
    ))
    return(uv)
  }
}

# TODO: we should pass --cache-dir=file.path(rappdirs::user_cache_dir("r-reticulate"), "uv-cache")
# if we are using a reticulate-managed uv installation.

get_or_create_venv <- function(requirements = NULL, python_version = "3.10", exclude_newer = NULL) {
  if (length(requirements)) {
    if (length(requirements) == 1 &&
      grepl("[/\\]", requirements) &&
      file.exists(requirements)) {
      # path to a requirements.txt
      requirements <- c("--with-requirements", maybe_shQuote(requirements))
    } else {
      requirements <- paste0("\"", requirements, "\"", collapse = ", ")
      requirements <- sprintf("# dependencies =[%s]", requirements)
    }
  }

  if (!is.null(python_version)) {
    has_const <- substr(python_version, 1, 1) %in% c(">", "<", "=", "!")
    python_version[!has_const] <- paste0("==", python_version[!has_const])
    python_version <- paste0(python_version, collapse = ",")
    python_version <- sprintf("# requires-python = \"%s\"", python_version)
  }

  if (!is.null(exclude_newer)) {
    # todo, accept a POSIXct/lt, format correctly
    exclude_newer <- c("--exclude-newer", maybe_shQuote(exclude_newer))
  }

  outfile <- tempfile(fileext = ".txt")
  on.exit(unlink(outfile), add = TRUE)
  input <- paste(
    "# /// script",
    python_version,
    requirements,
    "# ///",
    "import sys",
    sprintf("with open('%s', 'w') as f:", outfile),
    "  print(sys.executable, file=f)",
    sep = "\n"
  )
  result <- suppressWarnings(system2(
    uv_binary(),
    c(
      "run",
      "--color=always",
      "--no-project",
      "--python-preference=only-managed",
      exclude_newer,
      "-"
    ),
    env = c(
      "UV_NO_CONFIG=1"
    ),
    stdout = TRUE, stderr = TRUE,
    input = input
  ))


  if (!is.null(attr(result, "status"))) {
    uv_error_msg <- sub(
      "No solution found when resolving `--with` dependencies:",
      "No solution found when resolving dependencies:",
      result,
      fixed = TRUE
    )

    # write out uv error message separately, since stop() might truncate.
    writeLines(uv_error_msg, stderr())
    msg <- c(
      "Python requirements could not be satisfied.",
      "Call `py_require()` to omit or replace conflicting requirements."
    )

    msg <- paste0(msg, collapse = "\n")
    stop(msg)
  }

  # result
  readLines(outfile)
}
