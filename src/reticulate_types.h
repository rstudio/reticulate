
#ifndef RETICULATE_TYPES_H
#define RETICULATE_TYPES_H

#include "libpython.h"
using namespace reticulate::libpython;

#define RCPP_NO_MODULES
#define RCPP_NO_SUGAR
#include <Rcpp.h>

inline void python_object_finalize(SEXP object);

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

extern bool s_is_python_initialized;

class GILScope {
 private:
  PyGILState_STATE gstate;
  bool acquired = false;

 public:
  GILScope() {
    if (s_is_python_initialized) {
      gstate = PyGILState_Ensure();
      acquired = true;
    }
  }

  GILScope(bool force) {
    if (force) {
      gstate = PyGILState_Ensure();
      acquired = true;
    }
  }

  ~GILScope() {
    if (acquired) PyGILState_Release(gstate);
  }
};

inline void python_object_finalize(SEXP object) {
  GILScope gilscope;
  PyObject* pyObject = (PyObject*)R_ExternalPtrAddr(object);
  if (pyObject != NULL)
    Py_DecRef(pyObject);
}

// define a PythonException struct that we can use to throw an
// exception from C++ code. The class contains an SEXP with an R condition
// object that can be used to generate an R error condition.
struct PythonException {
  SEXP condition;
  PythonException(SEXP condition_) : condition(condition_) {}
};


// This custom BEGIN_RCPP is effectively identical to upstream
// except for the last line, which we use to ensure that the
// GIL is acquired when calling into Python (and released otherwise).
// Limitations of the macro preprocessor make it difficult to
// do this in a more elegant way.

#undef BEGIN_RCPP
#define BEGIN_RCPP                           \
  int rcpp_output_type = 0;                  \
  int nprot = 0;                             \
  (void)rcpp_output_type;                    \
  SEXP rcpp_output_condition = R_NilValue;   \
  (void)rcpp_output_condition;               \
  static SEXP stop_sym = Rf_install("stop"); \
  try {                                      \
    GILScope gilscope;

// This custom END_RCPP is effectively identical to upstream
// except for the addition of one additional catch block, which
// we use to generate custom R conditions from Python exceptions.
// Limitations of the macro preprocessor make it difficult to
// do this in a more elegant way.

#undef END_RCPP
#define END_RCPP                                                            \
  }                                                                         \
  catch (PythonException & __ex__) {                                        \
    rcpp_output_type = 2;                                                   \
    rcpp_output_condition = __ex__.condition;                               \
  }                                                                         \
  catch (Rcpp::internal::InterruptedException & __ex__) {                   \
    rcpp_output_type = 1;                                                   \
  }                                                                         \
  catch (Rcpp::LongjumpException & __ex__) {                                \
    rcpp_output_type = 3;                                                   \
    rcpp_output_condition = __ex__.token;                                   \
  }                                                                         \
  catch (Rcpp::exception & __ex__) {                                        \
    rcpp_output_type = 2;                                                   \
    rcpp_output_condition = PROTECT(rcpp_exception_to_r_condition(__ex__)); \
    ++nprot;                                                                \
  }                                                                         \
  catch (std::exception & __ex__) {                                         \
    rcpp_output_type = 2;                                                   \
    rcpp_output_condition = PROTECT(exception_to_r_condition(__ex__));      \
    ++nprot;                                                                \
  }                                                                         \
  catch (...) {                                                             \
    rcpp_output_type = 2;                                                   \
    rcpp_output_condition =                                                 \
        PROTECT(string_to_try_error("c++ exception (unknown reason)"));     \
    ++nprot;                                                                \
  }                                                                         \
  if (rcpp_output_type == 1) {                                              \
    Rf_onintr();                                                            \
  }                                                                         \
  if (rcpp_output_type == 2) {                                              \
    SEXP expr = PROTECT(Rf_lang2(stop_sym, rcpp_output_condition));         \
    ++nprot;                                                                \
    Rf_eval(expr, R_BaseEnv);                                               \
  }                                                                         \
  if (rcpp_output_type == 3) {                                              \
    Rcpp::internal::resumeJump(rcpp_output_condition);                      \
  }                                                                         \
  UNPROTECT(nprot);                                                         \
  return R_NilValue;

#endif // RETICULATE_TYPES_H
