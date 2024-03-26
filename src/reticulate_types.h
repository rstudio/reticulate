
#ifndef RETICULATE_TYPES_H
#define RETICULATE_TYPES_H

#include "libpython.h"
using namespace reticulate::libpython;

#define RCPP_NO_MODULES
#define RCPP_NO_SUGAR
#include <Rcpp.h>

inline void python_object_finalize(SEXP object);
SEXP py_callable_as_function(SEXP refenv, bool convert);
SEXP py_class_names(PyObject*);
SEXP py_to_r_wrapper(SEXP ref);
bool is_py_object(SEXP x);
bool try_py_resolve_module_proxy(SEXP);
SEXP new_refenv();

extern SEXP sym_py_object;
extern SEXP sym_convert;
extern SEXP sym_simple;
extern SEXP sym_pyobj;



class PyObjectRef: public Rcpp::RObject {

public:

  explicit PyObjectRef(SEXP object, bool check = true) : Rcpp::RObject(object) {
    if(!check) return;
    if(is_py_object(object)) return;
    Rcpp::stop("Expected a python object, received a %s",
               Rf_type2char(TYPEOF(object)));
  }

  explicit PyObjectRef(PyObject* object, bool convert, bool simple = true) {
    // this steals a reference to 'object'.
    // (i.e., we call Py_DecRef on it eventually, from the xtptr finalizer)
    SEXP xptr = PROTECT(R_MakeExternalPtr((void*) object, R_NilValue, R_NilValue));
    R_RegisterCFinalizer(xptr, python_object_finalize);

    SEXP refenv = PROTECT(new_refenv());
    Rf_defineVar(sym_pyobj, xptr, refenv);
    Rf_defineVar(sym_convert, Rf_ScalarLogical(convert), refenv);
    bool callable = PyCallable_Check(object);
    if (callable || !simple)
      Rf_defineVar(sym_simple, Rf_ScalarLogical(false), refenv);
    Rf_setAttrib(refenv, R_ClassSymbol, py_class_names(object));

    if (callable) {
      SEXP r_fn = PROTECT(py_callable_as_function(refenv, convert));
      r_fn = PROTECT(py_to_r_wrapper(r_fn));
      this->set__(r_fn); // PROTECT()
      UNPROTECT(4);
    } else {
      this->set__(refenv);
      UNPROTECT(2);
    }

  }

  void set(PyObject* object) {
    // used to populate delay_load module proxies
    SEXP refenv = get_refenv();
    SEXP xptr = PROTECT(R_MakeExternalPtr((void*) object, R_NilValue, R_NilValue));
    R_RegisterCFinalizer(xptr, python_object_finalize);
    Rf_defineVar(sym_pyobj, xptr, refenv);
    UNPROTECT(1);
  }

  PyObject* get() const {
    PyObject* obj = nullable_get();
    if (obj == NULL) {
      if(try_py_resolve_module_proxy(get_refenv()))
        return get(); // maybe resolved module proxy
      Rcpp::stop("Unable to access object (object is from previous session and is now invalid)");
    }
    return obj;
  }

  PyObject* nullable_get() const {

    SEXP xptr = Rf_findVarInFrame(get_refenv(), sym_pyobj);

    if(TYPEOF(xptr) == EXTPTRSXP) {
      return (PyObject*) R_ExternalPtrAddr(xptr);
    }

    // might be a (lazy) module_proxy
    if(xptr == R_UnboundValue) {
      return(NULL);
    }

    Rcpp::stop("malformed pyobj");
    return NULL; // unreachable, for compiler
  }

  SEXP get_refenv() const {

    SEXP sexp = this->get__();
    while(TYPEOF(sexp) == CLOSXP)
      sexp = Rf_getAttrib(sexp, sym_py_object);

    if (TYPEOF(sexp) != ENVSXP)
      Rcpp::stop("malformed py_object, has type %s", Rf_type2char(TYPEOF(sexp)));

    return sexp;
  }

  operator PyObject*() const {
    return get();
  }

  bool is_null_xptr() const {
    return nullable_get() == NULL;
  }

  bool convert() const {
    SEXP sexp = Rf_findVarInFrame(get_refenv(), sym_convert);

    if(TYPEOF(sexp) == LGLSXP)
      return (bool) Rf_asLogical(sexp);

    return true;
  }

  bool simple() const {
    SEXP sexp = Rf_findVarInFrame(get_refenv(), sym_simple);

    if(TYPEOF(sexp) == LGLSXP)
      return (bool) Rf_asLogical(sexp);

    return true;
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
