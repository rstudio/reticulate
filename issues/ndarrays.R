
# https://github.com/rstudio/reticulate/issues/9

library(reticulate)

np <- import("numpy")

# ndims has higher order dimensions first!

# may need to use this to traverse the array:

# void* PyArray_GetPtr(PyArrayObject* aobj, npy_intp* ind) 

# Return a pointer to the data of the ndarray, aobj, at the N-dimensional index
# given by the c-array, ind, (which must be at least aobj ->nd in size). You may
# want to typecast the returned pointer to the data type of the ndarray.




# create a numpy 3d array from an R array
a <- array(c(1:24), dim = c(2,3,4))
r_to_py(a)

# create a numpy 3d array via Python
py <- py_run_string("import numpy; a = numpy.arange(1,25).reshape(4,2,3)")
py_get_attr(py, "a")    
py$a



