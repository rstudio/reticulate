# Error requesting conflicting package versions

    Code
      r_session({
        library(reticulate)
        py_require("numpy<2")
        py_require("numpy>=2")
        import("numpy")
        py_config()
      })
    Output
      > library(reticulate)
      > py_require("numpy<2")
      > py_require("numpy>=2")
      > import("numpy")
        × No solution found when resolving tool dependencies:
        ╰─▶ Because you require numpy<2 and numpy>=2, we can conclude that your
            requirements are unsatisfiable.
      uv error code: 1
      -- Current requirements -------------------------------------------------
       Python:   3.12.xx (reticulate default)
       Packages: numpy, numpy<2, numpy>=2
      -------------------------------------------------------------------------
      Error in uv_get_or_create_env() : 
        Call `py_require()` to remove or replace conflicting requirements.
      Error: Installation of Python not found, Python bindings not loaded.
      See the Python "Order of Discovery" here: https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery.
      Execution halted
      ------- session end -------
      success: false
      exit_code: 1

# Setting py_require(python_version) after initializing Python 

    Code
      r_session({
        Sys.unsetenv("RETICULATE_PYTHON")
        Sys.setenv(RETICULATE_USE_MANAGED_VENV = "yes")
        pkg_py_require <- (function(...) reticulate::py_require(...))
        pkg_py_require <- rlang::zap_srcref(pkg_py_require)
        environment(pkg_py_require) <- asNamespace("stats")
        library(reticulate)
        py_require(python_version = ">=3.9", "pandas")
        py_require(python_version = ">=3.8,<3.14")
        py_require(python_version = "3.11")
        pkg_py_require(packages = c("pandas", "numpy"), python_version = ">=3.10")
        prefix <- import("sys")$prefix
        import("numpy")
        import("pandas")
        stopifnot(py_version() == "3.11")
        py_require(python_version = ">=3.9.1")
        py_require(python_version = ">=3.8.1,<3.14")
        py_require(python_version = "3.11")
        pkg_py_require(python_version = ">=3.10")
        py_require("numpy")
        py_require("pandas")
        py_require(c("numpy", "pandas"), action = "set")
        py_require(c("notexist"), action = "remove")
        try(import("requests"))
        py_require("requests")
        import("requests")
        prefix2 <- import("sys")$prefix
        stopifnot(prefix != prefix2)
        pkg_py_require(python_version = ">=3.12")
        pkg_py_require("pandas", action = "remove")
        try(py_require(exclude_newer = "2020-01-01"))
        try(py_require(python_version = ">=3.12"))
        try(py_require("pandas", action = "remove"))
      })
    Output
      > Sys.unsetenv("RETICULATE_PYTHON")
      > Sys.setenv(RETICULATE_USE_MANAGED_VENV = "yes")
      > pkg_py_require <- (function(...) reticulate::py_require(...))
      > pkg_py_require <- rlang::zap_srcref(pkg_py_require)
      > environment(pkg_py_require) <- asNamespace("stats")
      > library(reticulate)
      > py_require(python_version = ">=3.9", "pandas")
      > py_require(python_version = ">=3.8,<3.14")
      > py_require(python_version = "3.11")
      > pkg_py_require(packages = c("pandas", "numpy"), python_version = ">=3.10")
      > prefix <- import("sys")$prefix
      > import("numpy")
      Module(numpy)
      > import("pandas")
      Module(pandas)
      > stopifnot(py_version() == "3.11")
      > py_require(python_version = ">=3.9.xx")
      > py_require(python_version = ">=3.8.xx,<3.14")
      > py_require(python_version = "3.11")
      > pkg_py_require(python_version = ">=3.10")
      > py_require("numpy")
      > py_require("pandas")
      > py_require(c("numpy", "pandas"), action = "set")
      > py_require(c("notexist"), action = "remove")
      > try(import("requests"))
      Error in py_module_import(module, convert = convert) : 
        ModuleNotFoundError: No module named 'requests'
      Run `reticulate::py_last_error()` for details.
      > py_require("requests")
      > import("requests")
      Module(requests)
      > prefix2 <- import("sys")$prefix
      > stopifnot(prefix != prefix2)
      > pkg_py_require(python_version = ">=3.12")
      Warning message:
      In reticulate::py_require(...) :
        Python version requirements cannot be changed after Python has been initialized.
      * Python version request: '>=3.12' (from package:stats)
      * Python version initialized: '3.11.xx'
      > pkg_py_require("pandas", action = "remove")
      Warning message:
      In reticulate::py_require(...) :
        After Python has initialized, only `action = 'add'` is supported.
      > try(py_require(exclude_newer = "2020-01-01"))
      Error in py_require(exclude_newer = "2020-01-01") : 
        `exclude_newer` cannot be changed after Python has initialized.
      > try(py_require(python_version = ">=3.12"))
      Error in py_require(python_version = ">=3.12") : 
        Python version requirements cannot be changed after Python has been initialized.
      * Python version request: '>=3.12'
      * Python version initialized: '3.11.xx'
      > try(py_require("pandas", action = "remove"))
      Error in py_require("pandas", action = "remove") : 
        After Python has initialized, only `action = 'add'` is supported.
      > 
      ------- session end -------
      success: true
      exit_code: 0

# Error requesting a package that does not exists

    Code
      r_session(attach_namespace = TRUE, {
        py_require(c("pandas", "numpy", "notexists"))
        uv_get_or_create_env()
      })
    Output
      > py_require(c("pandas", "numpy", "notexists"))
      > uv_get_or_create_env()
        × No solution found when resolving tool dependencies:
        ╰─▶ Because notexists was not found in the package registry and you require
            notexists, we can conclude that your requirements are unsatisfiable.
      uv error code: 1
      -- Current requirements -------------------------------------------------
       Python:   3.12.xx (reticulate default)
       Packages: numpy, pandas, notexists
      -------------------------------------------------------------------------
      Error in uv_get_or_create_env() : 
        Call `py_require()` to remove or replace conflicting requirements.
      Execution halted
      ------- session end -------
      success: false
      exit_code: 1

# Error requesting conflicting Python versions

    Code
      r_session(attach_namespace = TRUE, {
        py_require(python_version = ">=3.10")
        py_require(python_version = "<3.10")
        uv_get_or_create_env()
      })
    Output
      > py_require(python_version = ">=3.10")
      > py_require(python_version = "<3.10")
      > uv_get_or_create_env()
      Error in resolve_python_version(constraints = python_version, uv = uv) : 
        Requested Python version constraints could not be satisfied.
        constraints: ">=3.10,<3.10"
      Hint: Call `py_require(python_version = <string>, action = "set")` to replace constraints.
      Available Python versions found: 3.11.xx ....
      Calls: uv_get_or_create_env -> resolve_python_version
      Execution halted
      ------- session end -------
      success: false
      exit_code: 1

# Simple tests

    Code
      r_session(attach_namespace = TRUE, {
        py_require("pandas")
        py_require("numpy==2")
        py_require()
      })
    Output
      > py_require("pandas")
      > py_require("numpy==2")
      > py_require()
      ══════════════════════════ Python requirements ══════════════════════════
      ── Current requirements ─────────────────────────────────────────────────
       Python:   [No Python version specified. Will default to '3.12.xx']
       Packages: numpy, pandas, numpy==2
      ── R package requests ───────────────────────────────────────────────────
      R package  Python packages                           Python version      
      reticulate numpy                                                         
      > 
      ------- session end -------
      success: true
      exit_code: 0

---

    Code
      r_session(attach_namespace = TRUE, {
        py_require("pandas")
        py_require("numpy==2")
        py_require("numpy==2", action = "remove")
        py_require()
      })
    Output
      > py_require("pandas")
      > py_require("numpy==2")
      > py_require("numpy==2", action = "remove")
      > py_require()
      ══════════════════════════ Python requirements ══════════════════════════
      ── Current requirements ─────────────────────────────────────────────────
       Python:   [No Python version specified. Will default to '3.12.xx']
       Packages: numpy, pandas
      ── R package requests ───────────────────────────────────────────────────
      R package  Python packages                           Python version      
      reticulate numpy                                                         
      > 
      ------- session end -------
      success: true
      exit_code: 0

---

    Code
      r_session(attach_namespace = TRUE, {
        py_require("pandas")
        py_require("numpy==2")
        py_require("numpy==2", action = "remove")
        py_require(exclude_newer = "1990-01-01")
        py_require()
      })
    Output
      > py_require("pandas")
      > py_require("numpy==2")
      > py_require("numpy==2", action = "remove")
      > py_require(exclude_newer = "1990-01-01")
      > py_require()
      ══════════════════════════ Python requirements ══════════════════════════
      ── Current requirements ─────────────────────────────────────────────────
       Python:   [No Python version specified. Will default to '3.12.xx']
       Packages: numpy, pandas
       Exclude:  Anything newer than 1990-01-01
      ── R package requests ───────────────────────────────────────────────────
      R package  Python packages                           Python version      
      reticulate numpy                                                         
      > 
      ------- session end -------
      success: true
      exit_code: 0

---

    Code
      r_session(attach_namespace = TRUE, {
        py_require("pandas")
        py_require("numpy==2")
        py_require("numpy==2", action = "remove")
        py_require(exclude_newer = "1990-01-01")
        py_require(python_version = c("<=3.11", ">=3.10"))
        py_require()
      })
    Output
      > py_require("pandas")
      > py_require("numpy==2")
      > py_require("numpy==2", action = "remove")
      > py_require(exclude_newer = "1990-01-01")
      > py_require(python_version = c("<=3.11", ">=3.10"))
      > py_require()
      ══════════════════════════ Python requirements ══════════════════════════
      ── Current requirements ─────────────────────────────────────────────────
       Python:   <=3.11, >=3.10
       Packages: numpy, pandas
       Exclude:  Anything newer than 1990-01-01
      ── R package requests ───────────────────────────────────────────────────
      R package  Python packages                           Python version      
      reticulate numpy                                                         
      > 
      ------- session end -------
      success: true
      exit_code: 0

# Multiple py_require() calls from package are shows in one row

    Code
      r_session(attach_namespace = TRUE, {
        gr_package <- (function() {
          py_require(paste0("package", 1:20))
          py_require(paste0("package", 1:10), action = "remove")
          py_require(python_version = c("<=3.11", ">=3.10"))
        })
        environment(gr_package) <- asNamespace("graphics")
        gr_package()
        py_require()
      })
    Output
      > gr_package <- (function() {
      +     py_require(paste0("package", 1:20))
      +     py_require(paste0("package", 1:10), action = "remove")
      +     py_require(python_version = c("<=3.11", ">=3.10"))
      + })
      > environment(gr_package) <- asNamespace("graphics")
      > gr_package()
      > py_require()
      ══════════════════════════ Python requirements ══════════════════════════
      ── Current requirements ─────────────────────────────────────────────────
       Python:   <=3.11, >=3.10
       Packages: numpy, package11, package12, package13, package14,
                 package15, package16, package17, package18, package19,
                 package20
      ── R package requests ───────────────────────────────────────────────────
      R package  Python packages                           Python version      
      reticulate numpy                                                         
      graphics   package11, package12, package13,          <=3.11, >=3.10      
                 package14, package15, package16,                              
                 package17, package18, package19,                              
                 package20                                                     
      > 
      ------- session end -------
      success: true
      exit_code: 0

# py_require() standard library module

    Code
      r_session({
        library(reticulate)
        py_require("os")
        os <- import("os")
      })
    Output
      > library(reticulate)
      > py_require("os")
      > os <- import("os")
        × No solution found when resolving tool dependencies:
        ╰─▶ Because os was not found in the package registry and you require os, we
            can conclude that your requirements are unsatisfiable.
      uv error code: 1
      -- Current requirements -------------------------------------------------
       Python:   3.12.xx (reticulate default)
       Packages: numpy, os
      -------------------------------------------------------------------------
      Hint: `py_require()` expects Python package names rather than Python module names.
      Modules provided by the Python standard library such as `sys` and `os` should not be passed to `py_require()`.
      -------------------------------------------------------------------------
      Error in uv_get_or_create_env() : 
        Call `py_require()` to remove or replace conflicting requirements.
      Error: Installation of Python not found, Python bindings not loaded.
      See the Python "Order of Discovery" here: https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery.
      Execution halted
      ------- session end -------
      success: false
      exit_code: 1

