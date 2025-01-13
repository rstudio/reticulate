# Error requesting newer package version against an older snapshot

    Code
      r_session(attach_namespace = TRUE, {
        get_or_create_venv(c("numpy<2", "numpy>=2"))
      })
    Output
      > get_or_create_venv(c("numpy<2", "numpy>=2"))
        × No solution found when resolving `--with` dependencies:
        ╰─▶ Because you require numpy<2 and numpy>=2, we can conclude that your
            requirements are unsatisfiable.
      Error in get_or_create_venv(c("numpy<2", "numpy>=2")) : 
        Python requirements could not be satisfied.
      Python dependencies:  'numpy<2' 'numpy>=2'
      Call `py_require()` to remove or replace conflicting requirements.
      Execution halted
      ------- session end -------
      success: false
      exit_code: 1

