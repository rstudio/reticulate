#include <Python.h>
#include <Rcpp.h>

void py_decref(PyObject* object);

typedef Rcpp::XPtr<PyObject, Rcpp::PreserveStorage, py_decref> PyObjectPtr;
