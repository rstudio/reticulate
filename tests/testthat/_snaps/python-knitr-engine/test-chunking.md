    library(reticulate)

prose

    y = [2, 4, 6]

    # Comment about len
    print(len(y))

    ## 3

    # This comment about type should be attached to the code below.
    print(type(y))

    ## <class 'list'>

prose

    # Same Example with Collapsed Chunk
    y = [2, 4, 6]

    # Note the output of this command should available immediately below it (like with R).
    print(len(y))
    ## 3

    # Comment about type should be attached to the code below (like R). 
    # The output from the previous command should not be part of this
    print(type(y))
    ## <class 'list'>

    # Comparative Code for R
    x = c(2, 4, 6)

    # Output of length() is given just after the command (as expected)
    print(length(x))

    ## [1] 3

    # Comment about class is attached to the code below (as expected)
    print(class(x))

    ## [1] "numeric"

    # Comparative Code for R
    x = c(2, 4, 6)

    # Output of length() is given just after the command (as expected)
    print(length(x))
    ## [1] 3

    # Comment about class is attached to the code below (as expected)
    print(class(x))
    ## [1] "numeric"
