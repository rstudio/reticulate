    bt <- reticulate::import_builtins()
    bt$print("Hello world")

    ## Hello world

In `jupyter_compat = FALSE`: we should only see the output of both
expressions.

    print(1)

    ## 1

    print(2)

    ## 2

In `jupyter_compat = TRUE`: we should only see the output of both
expressions.

    print(1)

    ## 1

    print(2)

    ## 2

In `jupyter_compat = FALSE`: we should only see the output of both
expressions.

    1 + 0

    ## 1

    1 + 1

    ## 2

`jupyter_compat = TRUE`: we should only see the output of the last
expression.

    1 + 0
    1 + 1

    ## 2
