
#ifndef RETICULATE_TYPES_H
#define RETICULATE_TYPES_H

#include "libpython.h"
using namespace reticulate::libpython;

#define RCPP_NO_MODULES
#define RCPP_NO_SUGAR
#include <Rcpp.h>

inline void python_object_finalize(SEXP object) {
  PyObject* pyObject = (PyObject*)R_ExternalPtrAddr(object);
  if (pyObject != NULL)
    Py_DecRef(pyObject);
}

class PyObjectRef : public Rcpp::Environment {

public:

  explicit PyObjectRef(SEXP object) : Rcpp::Environment(object) {}

  explicit PyObjectRef(PyObject* object, bool convert) :
      Rcpp::Environment(Rcpp::Environment::empty_env().new_child(false)) {
    set(object);
    assign("convert", convert);
  }

  PyObject* get() const {

    SEXP pyObject = getFromEnvironment("pyobj");
    if (pyObject != R_NilValue) {
      PyObject* obj = (PyObject*)R_ExternalPtrAddr(pyObject);
      if (obj != NULL)
        return obj;
    }

    Rcpp::stop("Unable to access object (object is from previous session and is now invalid)");
  }

  operator PyObject*() const {
    return get();
  }

  bool is_null_xptr() const {
    SEXP pyObject = getFromEnvironment("pyobj");
    if (pyObject == NULL)
      return true;
    else if (pyObject == R_NilValue)
      return true;
    else if ((PyObject*)R_ExternalPtrAddr(pyObject) == NULL)
      return true;
    else
      return false;
  }

  void set(PyObject* object) {
    Rcpp::RObject xptr = R_MakeExternalPtr((void*) object, R_NilValue, R_NilValue);
    R_RegisterCFinalizer(xptr, python_object_finalize);
    assign("pyobj", xptr);
  }

  bool convert() const {
    Rcpp::RObject pyObject = getFromEnvironment("convert");
    if (pyObject == R_NilValue)
      return true;
    else
      return Rcpp::as<bool>(pyObject);
  }


  SEXP getFromEnvironment(const std::string& name) const {
    return Rcpp::Environment::get(name);
  }

};

#endif // RETICULATE_TYPES_H
