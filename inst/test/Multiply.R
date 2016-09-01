



py_run_file("~/packages/tensorflow/inst/test/Multiply.py")
main = py_import("__main__")
m = main$Multiply()
m$bob



mat = matrix(c(1,2,3,4), nrow = 2, ncol = 2 )
m$printObject(mat)
m$printObject(45)
m$printObject(list(1,2,3))
m$printObject(list(a = 1, b = 2,c = 3))

nil <- m$returnNone()
nil
m$matrix_double
m$matrix_int
m$matrix_logical

