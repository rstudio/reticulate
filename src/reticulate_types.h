#ifndef __RETICULATE_TYPES__
#define __RETICULATE_TYPES__

#include "libpython.h"
using namespace libpython;

#include <Rcpp.h>

inline void python_object_finalize(SEXP object) {
  PyObject* pyObject = (PyObject*)R_ExternalPtrAddr(object);
  if (pyObject != NULL)
    Py_DecRef(pyObject);
}

class PyObjectRef : public Rcpp::Environment {

public:
  
  explicit PyObjectRef(SEXP object) : Rcpp::Environment(object) {}
  
  explicit PyObjectRef(PyObject* object) : 
      Rcpp::Environment(Rcpp::Environment::empty_env().new_child(false)) {
    set(object);
  }
  
  PyObject* get() const {
    SEXP pyObject = getFromEnvironment("pyobj");
    if (pyObject != R_NilValue)
      return (PyObject*)R_ExternalPtrAddr(pyObject);
    else
      return NULL;
  }
  
  operator PyObject*() const {
    return get();
  }
  
  void set(PyObject* object) {
    SEXP xptr = R_MakeExternalPtr((void*) object, R_NilValue, R_NilValue);
    R_RegisterCFinalizer(xptr, python_object_finalize);
    assign("pyobj", xptr);
  }
  
  SEXP getFromEnvironment(const std::string& name) const {
    return Rcpp::Environment::get(name);
  }
    
};



#endif // __RETICULATE_TYPES__
