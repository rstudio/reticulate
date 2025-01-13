#' @export
py_require <- function(packages = NULL,
                       python_version = NULL,
                       exclude_newer = NULL,
                       action = c("add", "remove", "set")) {
  action <- match.arg(action)

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
    requested_is_package = isNamespace(top_env),
    packages = packages,
    python_version = python_version,
    exclude_newer = exclude_newer,
    action = action
  )))
  .globals$python_requirements <- pr

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
  python_version <- x$python_version
  if (is.null(python_version)) {
    python_version <- "[No version of Python selected yet]"
  } else {
    python_version <- paste0(python_version, collapse = ", ")
  }
  cat("------------------ Python requirements ----------------------\n")
  cat("Current requirements ----------------------------------------\n")
  cat(" Packages:", packages, "\n")
  cat(" Python:  ", python_version, "\n")
  if (!is.null(x$exclude_newer)) {
    cat(" Exclude: Updates after", x$exclude_newer, "\n")
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
      requested_is_package = TRUE,
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

get_or_create_venv <- function(packages = get_python_reqs("packages"),
                               python_version = get_python_reqs("python_version"),
                               exclude_newer = get_python_reqs("exclude_newer")) {
  if (length(packages))
    packages <- as.vector(rbind("--with", maybe_shQuote(packages)))

  if (length(python_version))
    python_version <- c("--python", maybe_shQuote(paste0(python_version, collapse = ",")))

  if (!is.null(exclude_newer)) {
    # todo, accept a POSIXct/lt, format correctly
    exclude_newer <- c("--exclude-newer", maybe_shQuote(exclude_newer))
  }

  input <- "import sys; print(sys.executable);"

  result <- suppressWarnings(system2(
    uv_binary(), c(
      "run",
      "--no-project",
      "--python-preference=only-managed",
      exclude_newer,
      python_version,
      packages,
      "python",
      "-c",
      shQuote("import sys; print(sys.executable);")
    ), env = c(
      "UV_NO_CONFIG=1"
    ),
    stdout = TRUE,
  ))
  if (!is.null(attr(result, "status"))) {
    msg <- c(
      "Python requirements could not be satisfied.",
      if (!is.null(python_version))
        paste0("Python version: ", python_version[2]),
      if (!is.null(packages))
        # TODO: wrap+indent+un_shQuote python packages
        paste0(c("Python dependencies: ", matrix(packages, nrow = 2)[2, ]),
               collapse = " "),
      if (!is.null(exclude_newer))
        paste0("Exclude newer: ", exclude_newer[2]),
      "Call `py_require()` to remove or replace conflicting requirements."
    )
    msg <- paste0(msg, collapse = "\n")
    # TODO: check if `stop()` will truncate msg, fix-up hint if yes.
    stop(msg)
  }

  result
}
