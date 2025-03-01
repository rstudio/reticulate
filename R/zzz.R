#' Interact with the Python Main Module
#'
#' The `py` object provides a means for interacting
#' with the Python main session directly from \R. Python
#' objects accessed through `py` are automatically converted
#' into \R objects, and can be used with any other \R
#' functions as needed.
#'
#' @format An \R object acting as an interface to the
#'   Python main module.
#'
#' @export
"py"

.onLoad <- function(libname, pkgname) {

  main <- NULL
  makeActiveBinding("py", env = asNamespace(pkgname), function() {

    # return main module if already initialized
    if (!is.null(main))
      return(main)

    # attempt to initialize main
    if (is_python_initialized())
      main <<- import_main(convert = TRUE)

    # return value of main
    main

  })

  # register a callback auto-flushing Python output as appropriate
  addTaskCallback(function(...) {

    enabled <- getOption("reticulate.autoflush", default = TRUE)
    if (enabled && is_python_initialized())
      py_flush_output()

    TRUE
  })

  # on macOS, set the OPENBLAS environment variable if possible, as otherwise
  # numpy will complain that we're using the broken Accelerate BLAS
  #
  # https://github.com/numpy/numpy/issues/15947
  #
  # also set OMP_NUM_THREADS to avoid issues with mixing different OpenMP
  # run-times into the same process
  if (Sys.info()[["sysname"]] == "Darwin") {
    openblas <- Sys.getenv("OPENBLAS", unset = NA)
    if (is.na(openblas) && file.exists("/usr/local/opt/openblas")) {
      Sys.setenv(OPENBLAS = "/usr/local/opt/openblas")
      threads <- Sys.getenv("OMP_NUM_THREADS", unset = NA)
      if (is.na(threads))
        Sys.setenv(OMP_NUM_THREADS = "1")
    }
  }

  ## Register S3 method for suggested package
  s3_register <- asNamespace("rlang")$s3_register
  s3_register("pillar::type_sum", "python.builtin.object")

  if (is_positron()) {

    setHook("reticulate.onPyInit", function() {

      disable <- nzchar(Sys.getenv("_RETICULATE_POSITRON_DISABLE_VARIABLE_INSPECTORS_"))
      if (disable)
        return()

      # options("ark.testing" = TRUE)
      inspectors <- tryCatch(import_positron_ipykernel_inspectors(), error = function(e) NULL)
      if(is.null(inspectors)) {
        # warning("positron_ipykernel.inspectors could not be found, variables pane support for Python objects will be limited")
        return()
      }

      .globals$get_positron_variable_inspector <- inspectors$get_inspector

      .ark.register_method <- get(".ark.register_method", envir = globalenv())

      for (method_name in c(
        "ark_positron_variable_display_value",
        "ark_positron_variable_display_type",
        "ark_positron_variable_has_children",
        "ark_positron_variable_kind",
        "ark_positron_variable_get_child_at",
        "ark_positron_variable_get_children"
      )) {
        for (class_name in c("python.builtin.object",
                             "rpytools.ark_variables.ChildrenOverflow")) {
          method <- get0(paste0(method_name, ".", class_name))
          if (!is.null(method)) {
            .ark.register_method(method_name, class_name, method)
          }
        }
      }
    })
  }
}


# .onUnload <- function(libpath) {
#   py_finalize() # called from reg.finalizer(.globals) instead.
# }


import_positron_ipykernel_inspectors <- function() {
  if(!is_positron())
    return (NULL)

  # in 2025.03 release, inspectors module moved to here
  tryCatch({
    .ps.ui.executeCommand <- get(".ps.ui.executeCommand", globalenv())
    ipykernel_path <- .ps.ui.executeCommand("positron.reticulate.getIPykernelPath")
    inspectors <- import_from_path("positron.inspectors", path = dirname(ipykernel_path))
    return(inspectors)
  },
  error = function(e) NULL)

  # inspectors module pre-2025.03
  tryCatch({
    # https://github.com/posit-dev/positron/pull/5368
    .ps.ui.executeCommand <- get(".ps.ui.executeCommand", globalenv())
    ipykernel_path <- .ps.ui.executeCommand("positron.reticulate.getIPykernelPath")
    inspectors <- import_from_path("positron_ipykernel.inspectors",
                                   path = dirname(ipykernel_path))
    return(inspectors)
  },
  error = function(e) NULL)


  # hacky "usually work" fallbacks for finding the positron-python extension,
  # until ark+positron are updated and can reliably provide the canonical path
  # (i.e., until https://github.com/posit-dev/positron/pull/5368 is in the release build)

  # Try inspecting `_` env var. Only works in some contexts.
  x <- Sys.getenv("_")
  # x ==
  # on mac: "/Applications/Positron.app/Contents/Resources/app/extensions/positron-r/resources/ark/ark"
  if (grepl("positron-r", x, ignore.case = TRUE)) {
    inspectors_path <- list.files(
      path =  sub("positron-r.*$", "positron-python", x),
      pattern = "^inspectors.py$",
      recursive = TRUE, full.names = TRUE
    )
    # inspectors_path ==
    # on mac: "/Applications/Positron.app/Contents/Resources/app/extensions/positron-python/python_files/positron/positron_ipykernel/inspectors.py"
    if (length(inspectors_path) == 1) {
      return(import_from_path("positron_ipykernel.inspectors",
                              path = dirname(dirname(inspectors_path))))

    }
  }

  # vscode/positron-python will place itself on the PATH sometimes
  PATH <- Sys.getenv("PATH")
  PATH <- strsplit(PATH, .Platform$path.sep, fixed = TRUE)[[1L]]
  PATH <- grep("positron-python", PATH, value = TRUE, ignore.case = TRUE)
  # PATH ==
  # on mac: "/Applications/Positron.app/Contents/Resources/app/extensions/positron-python/python_files/deactivate/zsh"
  for (path in PATH) {
    inspectors_path <- list.files(
      path =  sub("^(.*positron-python)(.*)$", "\\1", path),
      pattern = "^inspectors.py$",
      recursive = TRUE, full.names = TRUE
    )
    if (length(inspectors_path) == 1) {
      return(import_from_path("positron_ipykernel.inspectors",
                              path = dirname(dirname(inspectors_path))))
    }
  }

  NULL
}
