#include <Python.h>
#include <Rcpp.h>

inline void py_object_finalize(PyObject* object) {
  if (object != NULL)
    ::Py_DecRef(object);
}

typedef Rcpp::XPtr<PyObject, Rcpp::PreserveStorage, py_object_finalize> PyObjectXPtr;
