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
       Packages: numpy, numpy<2, numpy>=2
      -------------------------------------------------------------------------
      Error: Call `py_require()` to remove or replace conflicting requirements.
      Execution halted
      ------- session end -------
      success: false
      exit_code: 1

# Error requesting newer package version against an older snapshot

    Code
      r_session(attach_namespace = TRUE, {
        py_require("tensorflow==2.18.*")
        py_require(exclude_newer = "2024-10-20")
        uv_get_or_create_env()
      })
    Output
      > py_require("tensorflow==2.18.*")
      > py_require(exclude_newer = "2024-10-20")
      > uv_get_or_create_env()
        × No solution found when resolving `--with` dependencies:
        ╰─▶ Because only tensorflow<2.18.dev0 is available and you require
            tensorflow>=2.18.dev0, we can conclude that your requirements are
            unsatisfiable.
      
            hint: `tensorflow` was requested with a pre-release marker (e.g.,
            tensorflow>=2.18.dev0), but pre-releases weren't enabled (try:
            `--prerelease=allow`)
      -- Current requirements -------------------------------------------------
       Packages: numpy, tensorflow==2.18.*
       Exclude:  Anything newer than 2024-10-20
      -------------------------------------------------------------------------
      Error: Call `py_require()` to remove or replace conflicting requirements.
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
       Packages: numpy, pandas, notexists
      -------------------------------------------------------------------------
      Error: Call `py_require()` to remove or replace conflicting requirements.
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
      error: No interpreter found for Python >=3.10, <3.10 in virtual environments or managed installations
      -- Current requirements -------------------------------------------------
       Python:   >=3.10, <3.10
       Packages: numpy
      -------------------------------------------------------------------------
      Error: Call `py_require()` to remove or replace conflicting requirements.
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
      ========================== Python requirements ========================== 
      -- Current requirements -------------------------------------------------
       Python:   [No Python version specified]
       Packages: numpy, pandas, numpy==2
      -- R package requests --------------------------------------------------- 
      R package  Python package(s)                         Python version      
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
      ========================== Python requirements ========================== 
      -- Current requirements -------------------------------------------------
       Python:   [No Python version specified]
       Packages: numpy, pandas
      -- R package requests --------------------------------------------------- 
      R package  Python package(s)                         Python version      
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
      ========================== Python requirements ========================== 
      -- Current requirements -------------------------------------------------
       Python:   [No Python version specified]
       Packages: numpy, pandas
       Exclude:  Anything newer than 1990-01-01
      -- R package requests --------------------------------------------------- 
      R package  Python package(s)                         Python version      
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
        py_require(python_version = c("3.11", ">=3.10"))
        py_require()
      })
    Output
      > py_require("pandas")
      > py_require("numpy==2")
      > py_require("numpy==2", action = "remove")
      > py_require(exclude_newer = "1990-01-01")
      > py_require(python_version = c("3.11", ">=3.10"))
      > py_require()
      ========================== Python requirements ========================== 
      -- Current requirements -------------------------------------------------
       Python:   3.11, >=3.10
       Packages: numpy, pandas
       Exclude:  Anything newer than 1990-01-01
      -- R package requests --------------------------------------------------- 
      R package  Python package(s)                         Python version      
      reticulate numpy                                                         
      > 
      ------- session end -------
      success: true
      exit_code: 0

