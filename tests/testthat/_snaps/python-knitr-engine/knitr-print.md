    bt <- reticulate::import_builtins()
    bt$print("Hello world")

    ## Hello world

## Print statements should always be captured

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

## Only outputs of last expressions are captured in jupyter compat

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

## One can disable outputs by using `;`

In `jupyter_compat = FALSE`: we should only see the print statement
output

    print("hello");

    ## hello

    1 + 0;
    1 + 1;

`jupyter_compat = TRUE`: we should only see the print statement output

    print("hello");

    ## hello

    1 + 0;
    1 + 1;

## `jupyter_compat` works with interleaved expressions

    print('first_stdout')

    ## first_stdout

    'first_expression'
    print('second_stdout')

    ## second_stdout

    'final_expression'

    ## 'final_expression'
