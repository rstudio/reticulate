# Error requesting conflicting package versions

    Code
      r_session(attach_namespace = TRUE, {
        py_require("numpy<2")
        py_require("numpy>=2")
        uv_get_or_create_env()
      })
    Output
      > py_require("numpy<2")
      > py_require("numpy>=2")
      > uv_get_or_create_env()
        × No solution found when resolving `--with` dependencies:
        ╰─▶ Because you require numpy<2 and numpy>=2, we can conclude that your
            requirements are unsatisfiable.
      -- Current requirements -------------------------------------------------
       Python:   3.11.11 (reticulate default)
       Packages: numpy, numpy<2, numpy>=2
      -------------------------------------------------------------------------
      Error in uv_get_or_create_env() : 
        Call `py_require()` to remove or replace conflicting requirements.
      Execution halted
      ------- session end -------
      success: false
      exit_code: 1

# Error requesting a package that does not exists

    Code
      r_session(attach_namespace = TRUE, {
        py_require(c("pandas", "numpy", "notexists"))
        uv_get_or_create_env()
      })
    Output
      > py_require(c("pandas", "numpy", "notexists"))
      > uv_get_or_create_env()
        × No solution found when resolving `--with` dependencies:
        ╰─▶ Because notexists was not found in the package registry and you require
            notexists, we can conclude that your requirements are unsatisfiable.
      -- Current requirements -------------------------------------------------
       Python:   3.11.11 (reticulate default)
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
      Error in resolve_python_version(constraints = python_version) : 
        Requested Python version constraints could not be satisfied.
        constraints: ">=3.10,<3.10"
      Hint: Call `py_require(python_version = <string>, action = "set")` to replace constraints.
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
       Python:   [No Python version specified. Will default to '3.11.11']
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
       Python:   [No Python version specified. Will default to '3.11.11']
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
       Python:   [No Python version specified. Will default to '3.11.11']
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

