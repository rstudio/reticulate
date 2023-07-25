    bt <- reticulate::import_builtins()
    bt$print("Hello world")

    ## Hello world

Both in `jupyter_compat = TRUE` and `jupyter_compat = FALSE` we should
see the results, because a `print` was called:

    print(1)

    ## 1

    print(2)

    ## 2

    print(1)
    print(2)

    ## 2

For the `jupyter_compat = FALSE` mode we should see the output of both
expressions. In `jupyter_compat`, we should only see the output for the
last expression.

    1 + 0

    ## 1

    1 + 1

    ## 2

    1 + 0
    1 + 1

    ## 2
