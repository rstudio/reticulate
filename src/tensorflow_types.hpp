#include <Python.h>

#include <Rcpp.h>


inline void decrementPyObject(PyObject* object) {
  if (object != NULL)
    ::Py_DecRef(object);
}

typedef Rcpp::XPtr<PyObject, Rcpp::PreserveStorage, decrementPyObject> PyObjectPtr;
