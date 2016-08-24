#include <Python.h>
#include <Rcpp.h>

void py_object_decref(PyObject* object);

typedef Rcpp::XPtr<PyObject, Rcpp::PreserveStorage, py_object_decref> PyObjectPtr;
