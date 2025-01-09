# Error requesting newer package version against an older snapshot

    Code
      get_or_create_venv()
    Condition
      [1m[33mError[39m in `get_or_create_venv()`:[22m
      [33m![39m Python requirements could not be satisfied.
      ✖ Requirements:
         Packages: tensorflow==2.18.*
         Python: 
         Exclude newer: 2024-10-20
      ✖ Output from 'uv':
      Reading inline script metadata from `stdin`
        × No solution found when resolving script dependencies:
        ╰─▶ Because only tensorflow<2.18.dev0 is available and you require
            tensorflow>=2.18.dev0, we can conclude that your requirements are
            unsatisfiable.
      
            hint: `tensorflow` was requested with a pre-release marker (e.g.,
            tensorflow>=2.18.dev0), but pre-releases weren't enabled (try:
            `--prerelease=allow`)

