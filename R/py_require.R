#' @export
py_require <- function(packages = NULL,
                       python_version = NULL,
                       action = c("add", "omit", "replace"),
                       silent = TRUE) {
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

  if (missing(packages) && missing(python_version)) {
    return(get_python_reqs())
  }

  action <- match.arg(action)

  err_packages <- NULL
  err_python <- NULL

  msg_packages <- NULL
  msg_python <- NULL

  final_packages <- NULL
  final_python <- NULL

  has_error <- FALSE

  if (!is.null(packages)) {
    req_packages <- get_python_reqs("packages")
    if (action %in% c("replace", "omit")) {
      for (pkg in packages) {
        pkg_name <- extract_name(pkg)
        if (action == "omit" && pkg_name != pkg) {
          matches <- pkg == req_packages
        } else {
          matches <- pkg_name == extract_name(req_packages)
        }
        if (any(matches)) {
          match_pkgs <- req_packages[matches]
          match_pkgs <- ennumerate_packages(match_pkgs)
          req_packages <- req_packages[!matches]
          if (action == "replace") {
            req_packages <- c(req_packages, pkg)
            msg_packages <- c(msg_packages, paste(
              "Replaced", match_pkgs, "with", sprintf("\"%s\"", pkg)
            ))
          } else {
            msg_packages <- c(msg_packages, paste("Ommiting", match_pkgs))
          }
        } else {
          has_error <- TRUE
          err_msg <- sprintf("\"%s\"", pkg)
          if (action == "replace" && pkg != pkg_name) {
            err_msg <- sprintf("%s(searched for: \"%s\")", err_msg, pkg_name)
          }
          err_packages <- c(err_packages, err_msg)
        }
      }
      final_packages <- req_packages
    } else {
      msg_packages <- paste("Added", ennumerate_packages(packages))
      final_packages <- unique(c(req_packages, packages))
    }
    if (length(err_packages) > 0) {
      err_packages <- c(
        "Could not match",
        ennumerate_packages(err_packages, FALSE)
      )
      if (action == "replace") {
        err_packages <- c(
          err_packages,
          "\nTip: Check spelling, or remove from your command, and try again"
        )
      }
      if (action == "omit") {
        err_packages <- c(
          err_packages,
          "\nTip: Remove from your command, and try again"
        )
      }
      err_packages <- paste0(err_packages, collapse = " ")
    }
  }

  if (!is.null(python_version)) {
    req_python_versions <- get_python_reqs("python_version")
    final_python <- switch(action,
      add = unique(c(python_version, req_python_versions)),
      omit = req_python_versions[req_python_versions != python_version],
      replace = python_version
    )
  }

  if (!silent) {
    if (has_error) {
      stop("\n", add_dash(c(err_packages, err_python)), "\n", call. = FALSE)
    } else {
      cat(add_dash(c(msg_packages, msg_python)), "\n")
    }
  }

  if (!has_error) {
    top_env <- topenv(parent.frame())

    new_history <- c(
      get_python_reqs("history"),
      list(list(
        requested_from = environmentName(top_env),
        requested_is_package = isNamespace(top_env),
        packages = packages,
        action = action,
        python_version = python_version
      ))
    )

    set_python_reqs(
      packages = final_packages,
      python_versions = final_python,
      history = new_history
    )
  }

  invisible()
}

extract_name <- function(x) {
  as.character(lapply(x, function(x) {
    # If it's a URL or path to a binary or source distribution
    # (e.g., .whl, .sdist), try to extract the name
    is_dist <- grepl("/", x) ||
      grepl("\\.(whl|tar\\.gz|.zip|.tgz)$", x)
    if (is_dist) {
      # Remove path or URL leading up to the file name
      x <- sub(".*/", "", x)
      # Remove everything after the first "-", which
      # by the spec should be the *distribution* name.
      x <- sub("-.*$", "", x)

      # a whl/tar.gz or other package format
      # should have name standardized already
      # with `-` substituted with `_` already.
      return(x)
    }
    # If it's a package name with a version
    # constraint, remove the version constraint
    x <- sub("[=<>].*$", "", x) # Remove ver constraints like `=`, `<`, `>`

    # If it's a package name with a modifier like
    # `tensorflow[and-cuda]`, remove the modifier
    x <- sub("\\[.*$", "", x) # Remove modifiers like `[and-cuda]`
    # standardize, replace "-" with "_"
    gsub("-", "_", x, fixed = TRUE)
  }))
}

# Message helpers --------------------------------------------------------------

add_dash <- function(x) {
  if (length(x) == 1) {
    dashed <- ""
    spaces <- ""
  } else {
    dashed <- "- "
    spaces <- "  "
  }
  x <- gsub("\n", paste0("\n", spaces), x)
  paste0(dashed, x, collapse = "\n")
}

ennumerate_packages <- function(x, add_quotes = TRUE) {
  out <- NULL
  len_x <- length(x)
  for (i in seq_along(x)) {
    i_x <- len_x - i
    if (i_x > 1) {
      join <- ", "
    } else if (i_x == 1) {
      join <- ", and "
    } else {
      join <- NULL
    }
    if (add_quotes) {
      xi <- sprintf("\"%s\"", x[i])
    } else {
      xi <- x[i]
    }
    out <- c(out, xi, join)
  }
  paste0(out, collapse = "")
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
    x = c("all", "python_versions", "packages", "history")) {
  pr <- .globals$python_requirements
  x <- match.arg(x)
  switch(x,
    all = pr,
    python_versions = pr$python_versions,
    packages = pr$packages,
    history = pr$history
  )
}

set_python_reqs <- function(
    python_versions = NULL,
    packages = NULL,
    history = NULL) {
  pr <- get_python_reqs("all")
  pr$python_versions <- python_versions %||% pr$python_versions
  pr$packages <- packages %||% pr$packages
  pr$history <- history %||% pr$history
  .globals$python_requirements <- pr
  get_python_reqs("all")
}

.globals$python_requirements <- structure(
  .Data = list(
    python_versions = c(),
    packages = c(),
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
      # character vector of package requirements
      requirements <- as.vector(rbind("--with", maybe_shQuote(requirements)))
    }
  }

  if (length(python_version)) {
    if (length(python_version) != 1) {
      has_const <- substr(python_version, 1, 1) %in% c(">", "<", "=", "!")
      python_version[!has_const] <- paste0("==", python_version[!has_const])
      python_version <- paste0(python_version, collapse = ",")
    }
    python_version <- c("--python", maybe_shQuote(python_version))
  }

  if (!is.null(exclude_newer)) {
    # todo, accept a POSIXct/lt, format correctly
    exclude_newer <- c("--exclude-newer", maybe_shQuote(exclude_newer))
  }

  outfile <- tempfile(fileext = ".txt")
  on.exit(unlink(outfile), add = TRUE)
  input <- sprintf("
import sys
with open('%s', 'w') as f:
    print(sys.executable, file=f)

", outfile)
  # input <- "import sys; print(sys.executable);"

  result <- suppressWarnings(system2t(
    uv_binary(),
    c(
      # "--verbose",
      "run",
      "--color=always",
      "--no-project",
      # "--python-preference=only-managed",
      python_version,
      requirements,
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
