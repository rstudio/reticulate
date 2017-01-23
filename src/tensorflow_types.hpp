#ifndef __TENSORFLOW_TYPES__
#define __TENSORFLOW_TYPES__

#include "libpython.hpp"
#include <Rcpp.h>

inline void python_object_finalize(_PyObject* object) {
  if (object != NULL)
    ::_Py_DecRef(object);
}

typedef Rcpp::XPtr<_PyObject, Rcpp::PreserveStorage, python_object_finalize>
                                                                  PyObjectXPtr;


#endif // __TENSORFLOW_TYPES__
