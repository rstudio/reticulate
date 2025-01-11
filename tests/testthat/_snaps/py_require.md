# Error requesting newer package version against an older snapshot

    Code
      r_session({
        reticulate:::get_or_create_venv(c("numpy<2", "numpy>=2"))
      })
    Output
      > reticulate:::get_or_create_venv(c("numpy<2", "numpy>=2"))
        × No solution found when resolving `--with` dependencies:
        ╰─▶ Because you require numpy<2 and numpy>=2, we can conclude that your
            requirements are unsatisfiable.
      Error in reticulate:::get_or_create_venv(c("numpy<2", "numpy>=2")) : 
        Python requirements could not be satisfied.
      Call `py_require()` to remove or replace conflicting requirements.
      Execution halted
      ------- session end -------
      success: false
      exit_code: 1

