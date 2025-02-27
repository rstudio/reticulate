
#include "libpython.h"

#define RCPP_NO_MODULES
#define RCPP_NO_SUGAR

#include <Rcpp.h>
using namespace Rcpp;

#include "signals.h"
#include "reticulate_types.h"
#include "common.h"

#include "event_loop.h"
#include "tinythread.h"
#include "pending_py_calls_notifier.h"

#include <fstream>
#include <time.h>

#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

using namespace reticulate::libpython;

int _Py_Check(PyObject* o) {
  // default impl we assign to some Python api functions until we've initialized Python;
  return 0;
}


PyGILState_STATE _initialize_python_and_PyGILState_Ensure() {
  Function initialize_python = Environment::namespace_env("reticulate")["ensure_python_initialized"];
  initialize_python();
  return PyGILState_Ensure();
}

SEXP sym_pyobj;
SEXP sym_py_object;
SEXP sym_simple;
SEXP sym_convert;

SEXP ns_reticulate;

SEXP r_func_py_filter_classes;
SEXP r_func_get_r_trace;
SEXP r_func_py_callable_as_function;
SEXP r_func_r_to_py;
SEXP r_func_py_to_r;
SEXP r_func_py_to_r_wrapper;

tthread::thread::id s_main_thread = 0;

// [[Rcpp::init]]
void reticulate_init(DllInfo *dll) {
  // before python is initialized, make these symbols safe to call (always return false)
  PyIter_Check = &_Py_Check;
  PyCallable_Check = &_Py_Check;
  PyGILState_Ensure = &_initialize_python_and_PyGILState_Ensure;
  // Py_MakePendingCallsRun = &_Py_Check;

  sym_py_object = Rf_install("py_object");
  sym_simple = Rf_install("simple");
  sym_convert = Rf_install("convert");
  sym_pyobj = Rf_install("pyobj");

  ns_reticulate = Rf_findVarInFrame(R_NamespaceRegistry, Rf_install("reticulate"));

  r_func_py_filter_classes = Rf_findVar(Rf_install("py_filter_classes"), ns_reticulate);
  r_func_py_callable_as_function = Rf_findVar(Rf_install("py_callable_as_function"), ns_reticulate);
  r_func_r_to_py = Rf_findVar(Rf_install("r_to_py"), ns_reticulate);
  r_func_py_to_r = Rf_findVar(Rf_install("py_to_r"), ns_reticulate);
  r_func_py_to_r_wrapper = Rf_findVar(Rf_install("py_to_r_wrapper"), ns_reticulate);
  r_func_get_r_trace = Rf_findVar(Rf_install("get_r_trace"), ns_reticulate);

  s_main_thread = tthread::this_thread::get_id();
}

inline
bool is_main_thread() {
  return s_main_thread == tthread::this_thread::get_id();
}

// track whether we are using python 3 (set during py_initialize)
bool s_isPython3 = false;

// [[Rcpp::export]]
bool is_python3() {
  return s_isPython3;
}

// track whether this is an interactive session
bool s_isInteractive = false;
bool is_interactive() {
  return s_isInteractive;
}

// track whether we have required numpy
std::string s_numpy_load_error;
bool haveNumPy() {
  return s_numpy_load_error.empty();
}

bool requireNumPy() {
  if (!haveNumPy())
    stop("Required version of NumPy not available: " + s_numpy_load_error);
  return true;
}

bool isPyArray(PyObject* object) {
  if (!haveNumPy()) return false;

  return PyArray_Check(object);
}

bool isPyArrayScalar(PyObject* object) {
  if (!haveNumPy()) return false;

  return PyArray_CheckScalar(object);
}

// static buffers for Py_SetProgramName / Py_SetPythonHome
std::string s_python;
std::wstring s_python_v3;
std::string s_pythonhome;
std::wstring s_pythonhome_v3;



// helper to convert std::string to std::wstring
std::wstring to_wstring(const std::string& str) {
  std::wstring ws = std::wstring(str.size(), L' ');
  ws.resize(std::mbstowcs(&ws[0], str.c_str(), str.size()));
  return ws;
}

// helper to convert std::wstring to std::string
std::string to_string(const std::wstring& ws) {
  int maxnchar = ws.size() * 4;
  char *buffer = (char*) malloc(sizeof(char) * maxnchar);
  int nchar = wcstombs(buffer, ws.c_str(), maxnchar);
  std::string s(buffer, nchar);
  free(buffer);
  return s;
}


// forward declare error handling utility
SEXP py_fetch_error(bool maybe_reuse_cached_r_trace = false);


const char *r_object_string = "r_object";

// wrap an R object in a longer-lived python object "capsule"
SEXP py_capsule_read(PyObject* capsule) {

  SEXP object = (SEXP) PyCapsule_GetPointer(capsule, r_object_string);
  if (object == NULL)
    throw PythonException(py_fetch_error());

  // Rcpp_precious_preserve() returns a cell of a doubly linked list
  // with the original object preserved in the cell TAG().
  return TAG(object);

}


int free_sexp(void* sexp) {
  // wrap Rcpp_precious_remove() to satisfy
  // Py_AddPendingCall() signature and return value requirements
  Rcpp_precious_remove((SEXP) sexp);
  return 0;
}

void Rcpp_precious_remove_main_thread(SEXP object) {
  if (is_main_thread()) {
    return Rcpp_precious_remove(object);
  }

  // #Py_AddPendingCall can fail sometimes, so we retry a few times
  const size_t wait_ms = 100;
  size_t waited_ms = 0;
  while (Py_AddPendingCall(free_sexp, object) != 0) {

    tthread::this_thread::sleep_for(tthread::chrono::milliseconds(wait_ms));

    // increment total wait time and print a warning every 60 seconds
    waited_ms += wait_ms;
    if ((waited_ms % 60000) == 0)
        PySys_WriteStderr("Waiting to schedule object finalizer on main R interpeter thread...\n");
    else if (waited_ms > 60000 * 2) {
        // if we've waited more than 2 minutes, something is wrong
        PySys_WriteStderr("Error: unable to register R object finalizer on main thread\n");
        return;
    }
  }
}

void py_capsule_free(PyObject* capsule) {

  SEXP object = (SEXP)PyCapsule_GetPointer(capsule, r_object_string);
  if (object == NULL)
    throw PythonException(py_fetch_error());

  // the R api access must be from the main thread
  Rcpp_precious_remove_main_thread(object);
}

PyObject* py_capsule_new(SEXP object) {

  if(TYPEOF(object) == EXTPTRSXP &&
     R_ExternalPtrAddr(object) == NULL)
      stop("Invalid pointer");

  // if object == R_NilValue, this is a no-op, R_NilValue is reflected back.
  object = Rcpp_precious_preserve(object);

  return PyCapsule_New((void *)object, r_object_string, py_capsule_free);

}

PyObject* py_get_attr(PyObject* object, const std::string& name) {

  PyObject* attr = PyObject_GetAttrString(object, name.c_str());
  if(attr == NULL)
    PyErr_Clear();
  return attr;

}

bool is_r_object_capsule(PyObject* capsule) {
  return PyCapsule_IsValid(capsule, r_object_string);
}

// helper class for ensuring decref of PyObject in the current scope
template <typename T>
class PyPtr {

public:

  // attach on creation, decref on destruction
  PyPtr()
    : object_(NULL)
  {
  }

  explicit PyPtr(T* object)
    : object_(object)
  {
  }

  virtual ~PyPtr()
  {
    if (object_ != NULL) {
      Py_DecRef((PyObject*) object_);
    }
  }

  operator T*() const
  {
    return object_;
  }

  T* get() const
  {
    return object_;
  }

  void assign(T* object)
  {
    object_ = object;
  }

  T* detach()
  {
    T* object = object_;
    object_ = NULL;
    return object;
  }

  bool is_null() const
  {
    return object_ == NULL;
  }

private:

  // prevent copying
  PyPtr(const PyPtr&);
  PyPtr& operator=(const PyPtr&);

  // underlying object
  T* object_;
};

typedef PyPtr<PyObject> PyObjectPtr;
typedef PyPtr<PyArray_Descr> PyArray_DescrPtr;

inline PyObject* PyUnicode_AsBytes(PyObject* str) {
  return PyUnicode_AsEncodedString(str, /* encoding = */ NULL, /* errors = */ "ignore");
  // encoding = NULL  is fastpath to "utf-8"
}

PyObject* as_python_str(const std::string& str);

std::string as_std_string(PyObject* str) {

  // conver to bytes if its unicode
  PyObjectPtr pStr;
  if (PyUnicode_Check(str) || isPyArrayScalar(str)) {
    str = PyUnicode_AsBytes(str);
    pStr.assign(str);
  }

  char* buffer;
  Py_ssize_t length;
  int res = is_python3() ?
    PyBytes_AsStringAndSize(str, &buffer, &length) :
    PyString_AsStringAndSize(str, &buffer, &length);
  if (res == -1)
    throw PythonException(py_fetch_error());

  return std::string(buffer, length);
}

#define as_utf8_r_string(str) Rcpp::String(as_std_string(str))

PyObject* as_python_str(SEXP strSEXP, bool handle_na=false) {
  if (handle_na && strSEXP == NA_STRING) {
    Py_IncRef(Py_None);
    return Py_None;
  }

  if (is_python3()) {
    // python3 doesn't have PyString and all strings are unicode so
    // make sure we get a unicode representation from R
    const char * value = Rf_translateCharUTF8(strSEXP);
    return PyUnicode_FromString(value);
  } else {
    const char * value = Rf_translateChar(strSEXP);
    return PyString_FromString(value);
  }
}

PyObject* as_python_str(const std::string& str) {
  if (is_python3()) {
    return PyUnicode_FromString(str.c_str());
  } else {
    return PyString_FromString(str.c_str());
  }
}

bool has_null_bytes(PyObject* str) {
  char* buffer;
  int res = PyString_AsStringAndSize(str, &buffer, NULL);
  if (res == -1) {
    py_fetch_error();
    return true;
  } else {
    return false;
  }
}

// helpers to narrow python array type to something convertable from R,
// guaranteed to return NPY_BOOL, NPY_LONG, NPY_DOUBLE, NPY_CDOUBLE, or NPY_VOID
// or -1 if it's unable to return one of these types.
int narrow_array_typenum(int typenum) {

  switch(typenum) {
  // logical
  case NPY_BOOL:
    typenum = NPY_BOOL;
    break;
    // integer
  case NPY_BYTE:
  case NPY_UBYTE:
  case NPY_SHORT:
  case NPY_USHORT:
  case NPY_INT:
    typenum = NPY_LONG;
    break;
    // double
  case NPY_UINT:
  case NPY_ULONG:
  case NPY_ULONGLONG:
  case NPY_LONG:
  case NPY_LONGLONG:
  case NPY_HALF:
  case NPY_FLOAT:
  case NPY_DOUBLE:
  case NPY_DATETIME: // needs some additional special handling
    typenum = NPY_DOUBLE;
    break;

    // complex
  case NPY_CFLOAT:
  case NPY_CDOUBLE:
    typenum = NPY_CDOUBLE;
    break;


    // string/object (leave these alone)
  case NPY_STRING:
  case NPY_UNICODE:
  case NPY_OBJECT:
  case NPY_VSTRING:
    break;

  // raw
  case NPY_VOID:
    break;

    // unsupported
  default:
    typenum = -1;
    break;
  }

  return typenum;
}

npy_intp PyArray_ITEMSIZE(PyArrayObject* array) {
  PyArray_Descr *descr = ((PyArrayObject_fields*)array)->descr;
  switch (PyArray_RUNTIME_VERSION) {
  case NPY_VERSION_2:
    return ((_PyArray_DescrNumPy2*)descr)->elsize;
  case NPY_VERSION_1:
    return ((_PyArray_DescrNumPy1*)descr)->elsize;
  default:
    return -1;
  }
}

int narrow_array_typenum(PyArrayObject* array) {
  return narrow_array_typenum(PyArray_TYPE(array));
}

int narrow_array_typenum(PyArray_Descr* descr) {
  return narrow_array_typenum(descr->type_num);
}

bool is_numpy_str(PyObject* x) {
  if (!isPyArrayScalar(x))
    return false; // ndarray or other, not string

  PyArray_DescrPtr descrPtr(PyArray_DescrFromScalar(x));
  int typenum = narrow_array_typenum(descrPtr);
  return (typenum == NPY_STRING || typenum == NPY_UNICODE);
}

bool is_python_str(PyObject* x) {

  if (PyUnicode_Check(x))
    return true;

  // python3 doesn't have PyString_* so mask it out (all strings in
  // python3 will get caught by PyUnicode_Check, we'll ignore
  // PyBytes entirely and let it remain a python object)
  else if (!is_python3() && PyString_Check(x) && !has_null_bytes(x))
    return true;

  else if (is_numpy_str(x))
    return true;

  else
    return false;
}

// check whether a PyObject is None
bool py_is_none(PyObject* object) {
  return object == Py_None;
}

// convenience wrapper for PyImport_Import
PyObject* py_import(const std::string& module) {
  PyObjectPtr module_str(as_python_str(module));
  return PyImport_Import(module_str);
}


class PyErrorScopeGuard {
private:
  PyObject *er_type, *er_value, *er_traceback;
  bool pending_restore;

public:
  PyErrorScopeGuard() {
    PyErr_Fetch(&er_type, &er_value, &er_traceback);
    pending_restore = true;
  }

  void release(bool restore = false) {
    if (restore)
      PyErr_Restore(er_type, er_value, er_traceback);
    pending_restore = false;
  }

  ~PyErrorScopeGuard() {
    if (pending_restore)
      PyErr_Restore(er_type, er_value, er_traceback);
  }
};
// copied directly from purrr; used to call rlang::trace_back() in
// py_fetch_error() in such a way that it doesn't introduce a new
// frame in returned traceback
SEXP current_env(void) {
  static SEXP call = []() {

    // `sys.frame(sys.nframe())` doesn't work because `sys.nframe()`
    // returns the number of the frame in which evaluation occurs. It
    // doesn't return the number of frames on the stack. So we'd need
    // to evaluate it in the last frame on the stack which is what we
    // are looking for to begin with. We use instead this workaround:
    // Call `sys.frame()` from a closure to push a new frame on the
    // stack, and use negative indexing to get the previous frame.
    SEXP fn = PROTECT(R_ParseEvalString("function() sys.frame(-1)", R_BaseEnv));
    SEXP call = Rf_lang1(fn);
    R_PreserveObject(call);

    UNPROTECT(1);
    return call;
  }();

  // Rf_PrintValue(get_r_trace(false, false));

  return Rf_eval(call, R_BaseEnv);
}

static inline
SEXP eval_call(SEXP r_func, SEXP arg) {
  RObject cl(Rf_lang2(r_func, arg));
  return Rcpp_fast_eval(cl, ns_reticulate);
}

static inline
SEXP eval_call_fast_unsafe(SEXP r_func, SEXP arg) {
  SEXP cl = PROTECT(Rf_lang2(r_func, arg));
  SEXP res = Rf_eval(cl, ns_reticulate);
  UNPROTECT(1);
  return res;
}

static inline
SEXP eval_call(SEXP r_func, SEXP arg1, SEXP arg2) {
  RObject cl(Rf_lang3(r_func, arg1, arg2));
  return Rcpp_fast_eval(cl, ns_reticulate);
}

// static inline
// SEXP eval_call(SEXP r_func, SEXP arg1, bool arg2) {
//   return eval_call(r_func, arg1, Rf_ScalarLogical(arg2));
// }

// static inline
// SEXP eval_call_in_userenv(SEXP r_func, SEXP arg) {
//   RObject cl(Rf_lang2(r_func, arg));
//   return Rcpp_fast_eval(cl, current_env());
//   // this sometimes returns the reticulate ns env
//   // we need a new func, current_user_env(), that walks the frames, skipping reticulate ns frames.
// }

// static inline
// SEXP eval_call_in_userenv(SEXP r_func, SEXP arg1, SEXP arg2) {
//   SEXP cl = Rf_lang3(r_func, arg1, arg2);
//   RObject cl_(cl); // protect
//   return Rcpp_fast_eval(cl, current_env());
// }


bool s_is_python_initialized = false;
bool s_was_python_initialized_by_reticulate = false;

// [[Rcpp::export]]
bool was_python_initialized_by_reticulate() {
  return s_was_python_initialized_by_reticulate;
}

namespace {
const std::string PYTHON_BUILTIN = "python.builtin";
const std::string UNRESOLVABLE_NAME = "<missing-python-type-name>";

class ScopedAssign {
    bool& flag;
    bool oldValue;
public:
    explicit ScopedAssign(bool* f, bool newValue) : flag(*f), oldValue(*f) {
        flag = newValue;
    }
    ~ScopedAssign() {
        flag = oldValue;
    }
};

std::string get_module_name(PyObject* classPtr) {
    // Can't throw exceptions here, since we call this while unwinding due to an exception.
    PyObject* moduleObj;
    switch (PyObject_GetOptionalAttrString(classPtr, "__module__", &moduleObj)) {
    case 1: break;
    case 0: return "";
    case -1:
      // REprintf("fetching __module__ raised exception\n");
      // if (PyErr_Occurred()) PyErr_Print();
      PyErr_Clear();
      return "";
    }

    PyObjectPtr modulePtr(moduleObj);
    if (PyUnicode_Check(moduleObj)) {
      const char* moduleStr = PyUnicode_AsUTF8(moduleObj);
      if (moduleStr == NULL) {
        // if (PyErr_Occurred()) PyErr_Print();
        // REprintf("as_r_class(): failed to convert __module__ unicode object to string\n");
        PyErr_Clear();
        return "";
      }
      if (strcmp(moduleStr, "builtins") == 0) {
        return PYTHON_BUILTIN;
      } else {
        std::string module(moduleStr);
        return module;
      }
    }

    if (PyBytes_Check(moduleObj)) {
      // I'm pretty sure this only happened in Python 2, but we keep the check in case not.
      Py_ssize_t size;
      char* moduleStr;
      if (PyBytes_AsStringAndSize(moduleObj, &moduleStr, &size) == 0) {
        if (strcmp(moduleStr, "__builtin__") == 0) {
          return PYTHON_BUILTIN;
        }
        std::string module(moduleStr, size);
        return module;
      }
      if (PyErr_Occurred()) PyErr_Print();
      REprintf("as_r_class: failed to convert __module__ bytes object to string\n");
      return NULL;
    }

    // Fallback, if type(class) != type (i.e., it's a metaclass),
    // try to to fetch type(class).__module__. But make sure we're not recursing more than once
    static bool recursing = false;
    if (!recursing && !PyType_CheckExact(classPtr)) {
      auto _recursion_guard = ScopedAssign(&recursing, true);
      return get_module_name((PyObject*) Py_TYPE(classPtr));
    }

    // if (PyErr_Occurred()) PyErr_Print();
    // REprintf("__module__ not a string\n");
    // fallback for when __module__ is not a python string, or resolvable
    // from type(cls).__module__.
    // Note, we don't want to throw an exception here, as this is a hot code path
    // excercised heavily when already handling exceptions.
    return "";
}

std::string get_class_name(PyObject* classPtr) {
    // Can't throw exceptions here, since we call this while unwinding due to an exception.
    PyObject* nameObj;
    switch (PyObject_GetOptionalAttrString(classPtr, "__name__", &nameObj)) {
    case 1: break;
    case 0: return UNRESOLVABLE_NAME;
    case -1:
        // REprintf("fetching __name__ raised exception\n");
        // if (PyErr_Occurred()) PyErr_Print();
        PyErr_Clear();
        return UNRESOLVABLE_NAME;
    }

    PyObjectPtr namePtr(nameObj);
    if (PyUnicode_Check(nameObj)) {
        const char* nameStr = PyUnicode_AsUTF8(nameObj);
        if (nameStr == NULL) {
            // if (PyErr_Occurred()) PyErr_Print();
            // REprintf("as_r_class(): failed to convert __name__ unicode object to string\n");
            PyErr_Clear();
            return UNRESOLVABLE_NAME;
        }
        std::string name(nameStr);
        return name;
    }

    // if (PyErr_Occurred()) PyErr_Print();
    // REprintf("__name__ not a string\n");
    PyErr_Clear();
    return UNRESOLVABLE_NAME;
}

}  // anonymous namespace

std::string as_r_class(PyObject* classObj) {
    std::string module(get_module_name(classObj));
    std::string name(get_class_name(classObj));

    return module.empty() ? name : module + '.' + name;
}

SEXP py_class_names(PyObject* object, bool exception) {

  // Py_TYPE() usually returns a borrowed reference to object.__class__
  // but can differ if __class__ was modified after the object was created.
  // (e.g., wrapt.ObjectProxy(dict()), as encountered in
  // tensorflow.python.trackable.data_structures._DictWrapper)
  // In CPython, the definition of Py_TYPE() changed in Python 3.10
  // from a macro with no return type to a inline static function returning PyTypeObject*.
  // for back compat, we continue to define Py_TYPE as a macro in reticulate/src/libpython.h

  // Note in Python 3.13, building the class character vector for
  // `wrapt.ObjectProxy()` instances broke/changed yet again. Attribute access and
  // metaclass construction in Python 3.13 has changed, because support for
  // classmethod descriptors was removed. In Python 3.13, when iterating
  // over [cls.__module__ in inspect.getmro(type(obj))], the __module__
  // objects are `property` objects that can't be evaluated (not bound to an
  // instance, and, evaluating them anyway with property.fget(obj) raises
  // an exception.
  //
  // It seems that in 3.13, when defining a class that subclasses a class that
  // uses a metaclass, like `class Foo(wrapt.ObjectProxy): pass`,
  // there is then no way to get back the actual Foo.__module__, only the
  // module name of the proxied object instance, with the appropriate gymnastics.
  //
  // TensorFlow 2.18 does not yet support Python 3.13, and hopefully this will
  // sort itself out upstream before we have to accomodate for it here.
  //
  // We might end up needing to compare if `type(obj) == obj.__class__`, and if not,
  // we will need to concat the class names derived from both. So that, given
  //   class Dict(wrapt.ObjectProxy): pass; d = Dict({})
  // Ideally, we generate an R class vector for d:
  //   __main__.Dict, wrapt.wrappers.ObjectProxy, python.builtin.dict, python.builtin.object
  // It's not clear this is possible yet without direct comparison of `id` to a reified `wrapt.ObjectProxy`,
  // (and special-casing support for wrapt.ObjectProxy, not metaclasses in general)
  // and doing that efficiently and robustly on this hot code path is not worth the effort yet.
  // Will wait for TF 2.19 and see if this sorts itself out upstream.

  PyObject* type = (PyObject*) Py_TYPE(object);
  if (type == NULL) {

    // this code path gets heavily excercised by py_fetch_error()
    // Something going wrong here, then py_fetch_error() will be of no help.
    // Fortunatly, an Exception here should be an exceedingly rare occurance.
    if (PyErr_Occurred()) PyErr_Print();
    Rcpp::stop("Unable to resolve PyObject type.");
  }

  // call inspect.getmro to get the class and it's bases in
  // method resolution order
  static PyObject* getmro = []() -> PyObject* {
    PyObjectPtr inspect(py_import("inspect"));
    if (inspect.is_null())
      throw PythonException(py_fetch_error());

    PyObject* getmro = PyObject_GetAttrString(inspect, "getmro");
    if (getmro == NULL)
      throw PythonException(py_fetch_error());

    return getmro;
  }();

  PyObjectPtr classes(PyObject_CallFunctionObjArgs(getmro, type, NULL));
  if (classes.is_null()) {
    if (PyErr_Occurred()) PyErr_Print();
    Rcpp::stop("Exception raised by 'inspect.getmro(<pyobj>)'; unable to build R 'class' attribute");
  }

  // start adding class names
  std::vector<std::string> classNames;

  Py_ssize_t len = PyTuple_Size(classes);
  classNames.reserve(len+2);
  // +2 to possibly add python.builtin.object and python.builtin.iterator,
  // or "error" and "condition"

  // add the bases to the R class attribute
  for (Py_ssize_t i = 0; i < len; i++) {
    PyObject* base = PyTuple_GetItem(classes, i); // borrowed
    classNames.push_back(as_r_class(base));
  }

  // add python.builtin.object if we don't already have it
  if (classNames.empty() || classNames.back() != "python.builtin.object") {
    // typically already there for exceptions (most objects, actually)
    classNames.push_back("python.builtin.object");
  }

  // if it's an iterator, include python.builtin.iterator, before python.builtin.object
  if(PyIter_Check(object))
    classNames.insert(classNames.end() - 1, "python.builtin.iterator");

  // if it's a BaseException instance, append "error"/"interrupt" and "condition"
  if (exception) {
    if (PyErr_GivenExceptionMatches(type, PyExc_KeyboardInterrupt))
      classNames.push_back("interrupt");
    else
      classNames.push_back("error");
    classNames.push_back("condition");
  }

  RObject classNames_robj = Rcpp::wrap(classNames); // convert + protect
  RObject out = eval_call(r_func_py_filter_classes, (SEXP) classNames_robj);
  return out;
}


SEXP py_class_names(PyObject* object) {
  return py_class_names(object, (bool) PyExceptionInstance_Check(object));
}


// needs to be defined here, though only used in reticulate_types.h
SEXP new_refenv() {

#if defined(R_VERSION) && R_VERSION >= R_Version(4, 1, 0)
  return R_NewEnv(/*enclos =*/ R_EmptyEnv, /*hash =*/ false, /*size =*/ 0);
#else
  // R_NewEnv() C func introducted in R 4.1.
  // Prior to that, we need to call R func new.env()
  static SEXP call = []() {
    SEXP call = Rf_lang3(Rf_findFun(Rf_install("new.env"), R_BaseEnv),
                    /*hash =*/ Rf_ScalarLogical(FALSE),
                    /*parent =*/ R_EmptyEnv);
    R_PreserveObject(call);
    return call;
  }();
  return Rf_eval(call, R_BaseEnv);
#endif
}


// wrap a PyObject
// this steals a reference
PyObjectRef py_ref(PyObject* object, bool convert)
{

  // wrap
  PyObjectRef ref(object, convert);
  return ref;

}

static inline
bool inherits2(SEXP object, const char* name) {
  // like inherits in R, but iterates over the class STRSXP vector
  // in reverse, since python.builtin.object is typically at the tail.
  SEXP klass = Rf_getAttrib(object, R_ClassSymbol);
  if (TYPEOF(klass) == STRSXP) {
    for (int i = Rf_length(klass)-1; i >= 0; i--) {
      if (strcmp(CHAR(STRING_ELT(klass, i)), name) == 0)
        return true;
    }
  }
  return false;
}

bool inherits2(SEXP object, const char* name1, const char* name2) {
  // like inherits in R, but iterates over the class STRSXP vector
  // in reverse, since python.builtin.object is typically at the tail.
  SEXP klass = Rf_getAttrib(object, R_ClassSymbol);
  if (TYPEOF(klass) == STRSXP) {

    int i = Rf_length(klass)-1;

    for (; i >= 0; i--) {
      if (strcmp(CHAR(STRING_ELT(klass, i)), name2) == 0) {
        // found name2, now look for name1
        for (i--; i >= 0; i--)
          if (strcmp(CHAR(STRING_ELT(klass, i)), name1) == 0)
            return true; // found name1 also
        break; // did not find name1
      }
    }
  }
  return false;
}

//' Check if a Python object is a null externalptr
//'
//' @param x Python object
//'
//' @return Logical indicating whether the object is a null externalptr
//'
//' @details When Python objects are serialized within a persisted R
//'  environment (e.g. .RData file) they are deserialized into null
//'  externalptr objects (since the Python session they were originally
//'  connected to no longer exists). This function allows you to safely
//'  check whether whether a Python object is a null externalptr.
//'
//'  The `py_validate` function is a convenience function which calls
//'  `py_is_null_xptr` and throws an error in the case that the xptr
//'  is `NULL`.
//'
//' @export
// [[Rcpp::export]]
bool py_is_null_xptr(PyObjectRef x) {
  return x.is_null_xptr();
}

//' @rdname py_is_null_xptr
//' @export
// [[Rcpp::export]]
void py_validate_xptr(PyObjectRef x)
{
    if (!x.is_null_xptr())
        return;

    if (inherits2(x, "python.builtin.module"))
    {
        if (try_py_resolve_module_proxy(x.get_refenv()))
            if (!x.is_null_xptr())
                return;
    }

    stop("Object is a null externalptr (it may have been disconnected from "
         "the session where it was created)");
}

bool option_is_true(const std::string& name) {
  SEXP valueSEXP = Rf_GetOption(Rf_install(name.c_str()), R_BaseEnv);
  return Rf_isLogical(valueSEXP) && (as<bool>(valueSEXP) == true);
}

bool traceback_enabled() {
  Environment pkgEnv = Environment::namespace_env("reticulate");
  Function func = pkgEnv["traceback_enabled"];
  return as<bool>(func());
}



SEXP get_current_call(void) {
  static SEXP call = []() {

    SEXP fn = PROTECT(R_ParseEvalString("function() sys.call(-1)", R_BaseEnv));

    SEXP call = Rf_lang1(fn);
    R_PreserveObject(call);

    UNPROTECT(1);
    return call;
  }();

  return Rf_eval(call, R_BaseEnv);
}

SEXP get_r_trace(bool maybe_use_cached = false) {
  // should this be eval_call_in_userenv()?
  return eval_call(r_func_get_r_trace,
                   Rf_ScalarLogical(maybe_use_cached),
                   /*trim_tail = */ Rf_ScalarInteger(true));
}

SEXP py_fetch_error(bool maybe_reuse_cached_r_trace) {

  if(!is_main_thread()) {
    GILScope _gil;
    PyErr_Print();
    PySys_WriteStderr("\nUnable to fetch R backtrace from Python thread\n"); // TODO:
    return R_NilValue;
  }

  PyObject *excType, *excValue, *excTraceback;
  PyErr_Fetch(&excType, &excValue, &excTraceback);  // we now own the PyObjects

  if (!excType) {
    Rcpp::stop("Unknown Python error.");
  }

  if (PyErr_GivenExceptionMatches(excType, PyExc_KeyboardInterrupt)) {
    // Technically, we can safely delete this if branch and let the
    // KeyboardInterrupt fall through the standard exception raising codepath.
    // Meaning, we can treat it as a regular Exception, augment it with a
    // traceback, and then signal it as an interrupt condition that also
    // inherits from "python.builtin.KeyBoardInterrupt" (signaled via
    // base::stop(<cond>) in the Rcpp wrapper).
    //
    // We intercept early here just to avoid the overhead.
    if (excTraceback) Py_DecRef(excTraceback);
    if (excValue) Py_DecRef(excValue);
    Py_DecRef(excType);

    throw Rcpp::internal::InterruptedException();
  }

  PyErr_NormalizeException(&excType, &excValue, &excTraceback);

  if (excTraceback != NULL && excValue != NULL && s_isPython3) {
    PyException_SetTraceback(excValue, excTraceback);
    Py_DecRef(excTraceback);
  }

  PyObjectPtr pExcType(excType);  // decref on exit

  switch (PyObject_HasAttrStringWithError(excValue, "call")) {
  case 0:   { // attr missing
    // check if this exception originated in python using the `raise from`
    // statement with an exception that we've already augmented with the full
    // r_trace. (or similarly, raised a new exception inside an `except:` block
    // while it is catching an Exception that contains an r_trace). If we find
    // r_trace/r_call in a __context__ Exception, pull them forward to this
    // topmost exception.
    PyObject *context = NULL, *r_call = NULL, *r_trace = NULL;
    PyObject *excValue_tmp = excValue;

    while ((context = PyObject_GetAttrString(excValue_tmp, "__context__"))) {
      if ((r_call = PyObject_GetAttrString(context, "call"))) {
          PyObject_SetAttrString(excValue, "call", r_call);
          Py_DecRef(r_call);
      }
      if ((r_trace = PyObject_GetAttrString(context, "trace"))) {
          PyObject_SetAttrString(excValue, "trace", r_trace);
          Py_DecRef(r_trace);
      }
      excValue_tmp = context;
      Py_DecRef(context);
      if(r_call || r_trace) {
        break;
      }
    }
  }
   case -1: // Exception raised when checking for attr
     PyErr_Clear();
   case 1: // has attr
     break;
  }



  // make sure the exception object has some some attrs: call, trace
  switch (PyObject_HasAttrStringWithError(excValue, "trace")) {
  case 0: { // attr missing
    SEXP r_trace = PROTECT(get_r_trace(maybe_reuse_cached_r_trace));
    PyObject* r_trace_capsule(py_capsule_new(r_trace));
    PyObject_SetAttrString(excValue, "trace", r_trace_capsule);
    Py_DecRef(r_trace_capsule);
    UNPROTECT(1);
  }
  case -1: // Exception raised when checking for attr
    PyErr_Clear();
  case 1: // has attr
    break;
  }

  // Otherwise, try to capture the current call.

  // A first draft of this tried using: SEXP r_call = get_last_call();
  // with get_last_call() defined in Rcpp headers. Unfortunately, that would
  // skip over the actual call of interest, and frequently return NULL
  // for shallow call stacks. So we fetch the call directly
  // using the R API.
  switch (PyObject_HasAttrStringWithError(excValue, "call")) {
  case 0: {  // attr present
    // Technically we don't need to protect call, since
    // it would already be protected by it's inclusion in the R callstack,
    // but rchk flags it anyway, and so ...
    RObject r_call( get_current_call() );
    PyObject* r_call_capsule(py_capsule_new(r_call));
    PyObject_SetAttrString(excValue, "call", r_call_capsule);
    Py_DecRef(r_call_capsule);
  }
  case -1: // Exception raised when checking for attr
    PyErr_Clear();
  case 1: // has attr
    break;
  }


  // get the cppstack, r_cppstack
  // FIXME: this doesn't seem to work, always returns NULL
  // SEXP r_cppstack = PROTECT(rcpp_get_stack_trace());
  // PyObject* r_cppstack_capsule(py_capsule_new(r_cppstack));
  // UNPROTECT(1);
  // PyObject_SetAttrString(excValue, "r_cppstack", r_cppstack_capsule);
  // Py_DecRef(r_cppstack_capsule);

  PyObjectRef cond(excValue, true);

  static SEXP sym_py_last_exception = Rf_install("py_last_exception");
  static SEXP pkg_globals = Rf_eval(Rf_install(".globals"), ns_reticulate); // eval to force PROMSXP

  Rf_defineVar(sym_py_last_exception, cond, pkg_globals);

  if (flush_std_buffers() == -1)
    warning(
        "Error encountered when flushing python buffers sys.stderr and "
        "sys.stdout");

  return cond;
}

// [[Rcpp::export]]
SEXP py_flush_output() {
  if(s_is_python_initialized) {
    GILScope _gil;
    flush_std_buffers();
  }
  return R_NilValue;
}


class PyFlushOutputOnScopeExit {
  public:
    ~PyFlushOutputOnScopeExit() {
      if (flush_std_buffers() == -1)
        warning(
          "Error encountered when flushing python buffers sys.stderr and "
          "sys.stdout");
  }
};



std::string conditionMessage_from_py_exception(PyObject* exc) {
  // invoke 'traceback.format_exception_only(<traceback>)'

  static PyObject* format_exception_only = []() {
    PyObjectPtr tb_module(py_import("traceback"));
    if (tb_module.is_null()) {
      PyErr_Print();
      Rcpp::stop("Failed to format Python Exception; could not import traceback module");
    }

    PyObject* format_exception_only = PyObject_GetAttrString(tb_module, "format_exception_only");

    if (format_exception_only == NULL) {
      PyErr_Print();
      Rcpp::stop("Failed to format Python Exception; could not get traceback.format_exception_only");
    }

    return format_exception_only;
  }();


  PyObjectPtr formatted(PyObject_CallFunctionObjArgs(
      format_exception_only, Py_TYPE(exc), exc, NULL));

  if (formatted.is_null()) {
    PyErr_Print();
    Rcpp::stop("Failed to format Python Exception; traceback.format_exception_only() raised an Exception");
  }

  // build error text
  std::ostringstream oss;

  // PyList_GetItem() returns a borrowed reference, no need to decref.
  for (Py_ssize_t i = 0, n = PyList_Size(formatted); i < n; i++)
    oss << as_std_string(PyList_GetItem(formatted, i));

  static std::string hint = []() {
    Environment pkg_env(Environment::namespace_env("reticulate"));
    Function hint_fn = pkg_env[".py_last_error_hint"];
    CharacterVector r_result = hint_fn();
    return Rcpp::as<std::string>(r_result[0]);
  }();

  oss << hint;
  std::string error = oss.str();

  SEXP max_msg_len_s = PROTECT(Rf_GetOption1(Rf_install("warning.length")));
  std::size_t max_msg_len(Rf_asInteger(max_msg_len_s));
  UNPROTECT(1);

  if (error.size() > max_msg_len) {
    // R has a modest byte size limit for error messages, default 1000, user
    // adjustable up to 8170. Error messages beyond the limit are silently
    // truncated. If the message will be truncated, we truncate it a little
    // better here and include a useful hint in the error message.

    std::string trunc("<...truncated...>");

    // TensorFlow since ~2.6 has been including a curated traceback as part of
    // the formatted exception message, with the most user-actionable content
    // towards the tail. Since the tail is the most useful part of the message,
    // truncate from the middle of the exception by default, after including the
    // first two lines.
    int over(error.size() - max_msg_len);
    int first_line_end_pos(error.find("\n"));
    int second_line_start_pos(error.find("\n", first_line_end_pos + 1));
    std::string head(error.substr(0, second_line_start_pos + 1));
    std::string tail(
        error.substr(over + head.size() + trunc.size() + 20,
                     std::string::npos));
    // +20 to accommodate "Error: " and similar accruals from R signal handlers.
    error = head + trunc + tail;
  }

  return error;
}

// [[Rcpp::export]]
std::string conditionMessage_from_py_exception(PyObjectRef exc) {
  GILScope _gil;
  return conditionMessage_from_py_exception(exc.get());
}

// check whether the PyObject can be mapped to an R scalar type
int r_scalar_type(PyObject* x) {

  if (PyBool_Check(x))
    return LGLSXP;

  // integer
  else if (PyInt_Check(x) || PyLong_Check(x))
    return INTSXP;

  // double
  else if (PyFloat_Check(x))
    return REALSXP;

  // complex
  else if (PyComplex_Check(x))
    return CPLXSXP;

  else if (is_python_str(x))
    return STRSXP;

  // not a scalar
  else
    return NILSXP;
}

// check whether the PyObject is a list of a single R scalar type
int scalar_list_type(PyObject* x) {

  Py_ssize_t len = PyList_Size(x);
  if (len == 0)
    return NILSXP;

  PyObject* first = PyList_GetItem(x, 0);
  int scalarType = r_scalar_type(first);
  if (scalarType == NILSXP)
    return NILSXP;

  for (Py_ssize_t i = 1; i<len; i++) {
    PyObject* next = PyList_GetItem(x, i);
    if (r_scalar_type(next) != scalarType)
      return NILSXP;
  }

  return scalarType;
}

bool py_equal(PyObject* x, const std::string& str) {

  PyObjectPtr pyStr(as_python_str(str));
  if (pyStr.is_null())
    throw PythonException(py_fetch_error());

  return PyObject_RichCompareBool(x, pyStr, Py_EQ) == 1;

}

bool is_pandas_na(PyObject* x) {

  // don't want to prematurely import pandas if we don't need it.
  // ideally we would be able to do something like this
  // static PyObject* pandas_NAType = NULL;
  // if (pandas_NAType == NULL) {
  //   PyObjectPtr inspect(py_import("pandas"));
  //   if (inspect.is_null())
  //     throw PythonException(py_fetch_error());
  // }
  //  PyObject* type = (PyObject*) Py_TYPE(object);
  // if (type == NULL)
  //   throw PythonException(py_fetch_error());

  // call inspect.getmro to get the class and it's bases in
  // method resolution order


  // retrieve class object
  PyObjectPtr pyClass(py_get_attr(x, "__class__"));
  if (pyClass.is_null())
    return false;

  PyObjectPtr pyModule(py_get_attr(pyClass, "__module__"));
  if (pyModule.is_null())
    return false;

  // check for expected module name
  if (!py_equal(pyModule, "pandas._libs.missing"))
    return false;

  // retrieve class name
  PyObjectPtr pyName(py_get_attr(pyClass, "__name__"));
  if (pyName.is_null())
    return false;

  // check for expected names
  return py_equal(pyName, "NAType") ||
    py_equal(pyName, "C_NAType");

}

#define STATIC_MODULE(module)                                      \
  const static PyObject* mod = PyImport_ImportModule(module);      \
  if (mod == NULL) {                                               \
    throw PythonException(py_fetch_error());                       \
  }                                                                \
  return const_cast<PyObject*>(mod);

PyObject* numpy () {
  STATIC_MODULE("numpy")
}

PyObject* pandas_arrays () {
  STATIC_MODULE("pandas.arrays")
}

bool is_pandas_na_like(PyObject* x) {
  const static PyObject* np_nan = PyObject_GetAttrString(numpy(), "nan");
  return is_pandas_na(x) || (x == Py_None) || (x == (PyObject*)np_nan);
}

void set_string_element(SEXP rArray, int i, PyObject* pyStr) {
  if (is_pandas_na_like(pyStr)) {
    SET_STRING_ELT(rArray, i, NA_STRING);
    return;
  }
  std::string str = as_std_string(pyStr);
  cetype_t ce = PyUnicode_Check(pyStr) ? CE_UTF8 : CE_NATIVE;
  SEXP strSEXP = Rf_mkCharCE(str.c_str(), ce);
  SET_STRING_ELT(rArray, i, strSEXP);
}

static inline
bool py_has_attr(PyObject* x_, const char* name) {
  switch (PyObject_HasAttrStringWithError(x_, name)) {
  case 1: return true;
  case 0: return false;
  case -1:
  default:
    PyErr_Clear();
    return false;
  }
}


bool py_is_callable(PyObject* x) {
  return PyCallable_Check(x) == 1 || py_has_attr(x, "__call__");
}

// [[Rcpp::export]]
PyObjectRef py_none_impl() {
  GILScope _gil;
  Py_IncRef(Py_None);
  return py_ref(Py_None, false);
}

// [[Rcpp::export]]
bool py_is_callable(PyObjectRef x) {
  if (x.is_null_xptr())
    return false;
  GILScope _gil;
  return py_is_callable(x.get());
}

// caches np.nditer function so we don't need to obtain it everytime we want to
// cast numpy string arrays into R objects.
PyObject* get_np_nditer() {
  static PyObject* np_nditer = []() -> PyObject* {
    PyObject* np_nditer = PyObject_GetAttrString(numpy(), "nditer");
    if (np_nditer == NULL) {
      throw PythonException(py_fetch_error());
    }
    return np_nditer;
  }();

  // Return the static np_nditer
  return np_nditer;
}


SEXP py_callable_as_function(SEXP callable, bool convert) {
  SEXP f = PROTECT(eval_call_fast_unsafe(r_func_py_callable_as_function, callable));

  // copy over class attribute
  Rf_setAttrib(f, R_ClassSymbol, Rf_getAttrib(callable, R_ClassSymbol));

  // save reference to underlying py_object
  Rf_setAttrib(f, sym_py_object, callable);

  UNPROTECT(1);
  return f;
}



SEXP py_to_r_wrapper(SEXP x) {
  SEXP x2 = eval_call(r_func_py_to_r_wrapper, x);
  if(x == x2) // no method, py_to_r_wrapper.default() reflects
    return(x);

  // copy over all attributes ("class" and "py_object", typically)
  // similar to Rf_copyMostAttrib(x, x2);, but copies *all* attribs
  PROTECT(x2);
  SEXP a = ATTRIB(x);
  while (a != R_NilValue) {
    Rf_setAttrib(x2, TAG(a), CAR(a));
    a = CDR(a);
  }
  SET_OBJECT(x2, 1);
  UNPROTECT(1);
  return x2;
}


SEXP py_to_r_cpp(PyObject* x, bool convert, bool simple = true);

//' Check if x is a Python object
//'
//' Checks if `x` is a Python object, more efficiently
//' than `inherits(x, "python.builtin.object")`.
//'
//' @param x An \R or Python.
//'
//' @return \code{TRUE} or \code{FALSE}.
//' @export
//' @keywords internal
// [[Rcpp::export]]
bool is_py_object(SEXP x) {
  if(OBJECT(x)) {
    switch (TYPEOF(x)) {
    case ENVSXP:
    case CLOSXP:
      return inherits2(x, "python.builtin.object");
    case VECSXP:
      return inherits2(x, "python.builtin.object", "condition");
    }
  }
  return false;
}

// convert a python object to an R object
// if we fail to convert, this creates a new reference to x
// (i.e, caller should call Py_DecRef() / PyObjectPtr.detach() for any refs caller created)
SEXP py_to_r(PyObject* x, bool convert) {
  GILScope _gil;
  if(!convert) {
    Py_IncRef(x);
    return py_ref(x, convert);
  }

  // first try the fast path
  SEXP result = py_to_r_cpp(x, true);
  if(!is_py_object(result))
    return(result);

  // ideally this would call in userenv, so UseMethod() finds methods there.
  return eval_call(r_func_py_to_r, result);
}


// [[Rcpp::export]]
SEXP py_to_r_cpp(SEXP x) {

  // reflect non python objects
  if (!is_py_object(x)) return x;

  PyObjectRef ref(x, /*check =*/false);
  bool simple = ref.simple();

  // if we already know this is not a simple py object,
  // and `convert` is already TRUE, there is nothing to do;
  // return x unmodified
  if(!simple && ref.convert()) return x;

  GILScope _gil;
  // if simple = true, call py_to_r_cpp(PyObject) to (try to) simplify
  // if convert = false, call py_to_r_cpp(PyObject) to get a new ref with convert = true
  SEXP ret = py_to_r_cpp(ref.get(), /*convert =*/ true, simple);
  if(simple && is_py_object(ret)) {
    PROTECT(ret);
    Rf_defineVar(sym_simple, Rf_ScalarLogical(FALSE), ref.get_refenv());
    UNPROTECT(1);
  }
    // if we tried and failed to simplify the ref,
    // mark the ref as non simple so we can skip trying next time.
  return ret; // return the new ref
}

// fast path
SEXP py_to_r_cpp(PyObject* x, bool convert, bool simple) {
  // object is assumed to be simple unless proven otherwise.
  // simple==false allows for skipping all the checks.
  // an object can pass through here multiple times during conversion
  // because we call it before py_to_r S3 dispatch, and also,
  // in py_to_r.default.
  // we do this for consistency between objects that originate in cpp
  // and that originate from R.
  if (convert && simple) {

  // NULL for Python None
  if (py_is_none(x))
    return R_NilValue;

  // check for scalars
  int scalarType = r_scalar_type(x);
  if (scalarType != NILSXP) {

    // logical
    if (scalarType == LGLSXP)
      return LogicalVector::create(x == Py_True);

    // integer
    else if (scalarType == INTSXP)
      return IntegerVector::create(PyInt_AsLong(x));

    // double
    else if (scalarType == REALSXP)
      return NumericVector::create(PyFloat_AsDouble(x));

    // complex
    else if (scalarType == CPLXSXP) {
      Rcomplex cplx;
      cplx.r = PyComplex_RealAsDouble(x);
      cplx.i = PyComplex_ImagAsDouble(x);
      return ComplexVector::create(cplx);
    }

    // string
    else if (scalarType == STRSXP)
      return CharacterVector::create(as_utf8_r_string(x));

    else
      return R_NilValue; // keep compiler happy
  }

  // list
  if (PyList_CheckExact(x)) {

    Py_ssize_t len = PyList_Size(x);
    int scalarType = scalar_list_type(x);
    if (scalarType == REALSXP) {
      Rcpp::NumericVector vec(len);
      for (Py_ssize_t i = 0; i<len; i++)
        vec[i] = PyFloat_AsDouble(PyList_GetItem(x, i));
      return vec;
    } else if (scalarType == INTSXP) {
      Rcpp::IntegerVector vec(len);
      for (Py_ssize_t i = 0; i<len; i++)
        vec[i] = PyInt_AsLong(PyList_GetItem(x, i));
      return vec;
    } else if (scalarType == CPLXSXP) {
      Rcpp::ComplexVector vec(len);
      for (Py_ssize_t i = 0; i<len; i++) {
        PyObject* item = PyList_GetItem(x, i);
        Rcomplex cplx;
        cplx.r = PyComplex_RealAsDouble(item);
        cplx.i = PyComplex_ImagAsDouble(item);
        vec[i] = cplx;
      }
      return vec;
    } else if (scalarType == LGLSXP) {
      Rcpp::LogicalVector vec(len);
      for (Py_ssize_t i = 0; i<len; i++)
        vec[i] = PyList_GetItem(x, i) == Py_True;
      return vec;
    } else if (scalarType == STRSXP) {
      Rcpp::CharacterVector vec(len);
      for (Py_ssize_t i = 0; i<len; i++)
        vec[i] = as_utf8_r_string(PyList_GetItem(x, i));
      return vec;
    } else { // not a homegenous list of scalars, return a list
      Rcpp::List list(len);
      for (Py_ssize_t i = 0; i<len; i++)
        list[i] = py_to_r(PyList_GetItem(x, i), convert); // borrowed ref
      return list;
    }
  }

  // tuple (but don't convert namedtuple as it's often a custom class)
  if (PyTuple_CheckExact(x) && !py_has_attr(x, "_fields")) {
    Py_ssize_t len = PyTuple_Size(x);
    Rcpp::List list(len);
    for (Py_ssize_t i = 0; i<len; i++)
      list[i] = py_to_r(PyTuple_GetItem(x, i), convert);
    return list;
  }

  // dict
  if (PyDict_CheckExact(x)) {
    // if you're tempted to change this to PyDict_Check() to allow subclasses, don't.
    // https://github.com/rstudio/reticulate/issues/1510
    // https://github.com/rstudio/reticulate/issues/1429#issuecomment-1658499679
    // https://github.com/rstudio/reticulate/issues/1360#issuecomment-1680413674


    // copy the dict and allocate
    PyObjectPtr dict(PyDict_Copy(x));
    Py_ssize_t size = PyDict_Size(dict);
    std::vector<std::string> names(size);
    Rcpp::List list(size);

    // iterate over dict
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    Py_ssize_t idx = 0;
    while (PyDict_Next(dict, &pos, &key, &value)) {
      if (is_python_str(key)) {
        names[idx] = as_utf8_r_string(key);
      } else {
        PyObjectPtr str(PyObject_Str(key));
        names[idx] = as_utf8_r_string(str);
      }
      list[idx] = py_to_r(value, convert);
      idx++;
    }
    list.names() = names;
    return list;

  }

  // numpy array
  if (isPyArray(x)) {

    // R array to return
    RObject rArray = R_NilValue;

    // get the array
    PyArrayObject* array = (PyArrayObject*) x;
    PyObjectPtr array_; // in case we need to decref before we're done;

    // get the dimensions -- treat 0-dim array (numpy scalar) as
    // a 1-dim for conversion to R (will end up with a single
    // element R vector)
    npy_intp len = PyArray_SIZE(array);
    int nd = PyArray_NDIM(array);
    Rcpp::IntegerVector dimsVector(nd);
    if (nd > 0) {
      npy_intp *dims = PyArray_DIMS(array);
      for (int i = 0; i<nd; i++)
        dimsVector[i] = dims[i];
    } else {
      dimsVector.push_back(1);
    }
    // determine the target type of the array
    int og_typenum = PyArray_TYPE(array);
    int typenum = narrow_array_typenum(og_typenum);
    if (typenum == -1) {
      simple = false;
      goto cant_convert;
    }

    if(og_typenum == NPY_DATETIME) {
      PyObjectPtr dtype_str(as_python_str("datetime64[ns]"));
      PyObject* array2 = PyObject_CallMethod(x, "astype", "O", (PyObject*) dtype_str.get()); // new ref
      array_.assign(array2); // decref new array on scope exit
      array = (PyArrayObject*) array2;
    }

    // cast it to a fortran array (PyArray_CastToType steals the descr)
    // (note that we will decref the copied array below)
    PyArray_Descr* descr = PyArray_DescrFromType(typenum);
    array = (PyArrayObject*) PyArray_CastToType(array, descr, NPY_ARRAY_FARRAY);
    if (array == NULL)
      throw PythonException(py_fetch_error());

    // ensure we release it within this scope
    PyObjectPtr ptrArray((PyObject*)array);

    // copy the data as required per-type
    switch(typenum) {

      case NPY_BOOL: {
        npy_bool* pData = (npy_bool*)PyArray_DATA(array);
        rArray = Rf_allocArray(LGLSXP, dimsVector);
        int* rArray_ptr = LOGICAL(rArray);
        for (int i=0; i<len; i++)
          rArray_ptr[i] = pData[i];
        break;
      }

      case NPY_LONG: {
        npy_long* pData = (npy_long*)PyArray_DATA(array);
        rArray = Rf_allocArray(INTSXP, dimsVector);
        int* rArray_ptr = INTEGER(rArray);
        for (int i=0; i<len; i++)
          rArray_ptr[i] = pData[i];
        break;
      }

      case NPY_DOUBLE: {
        npy_double* pData = (npy_double*)PyArray_DATA(array);
        rArray = Rf_allocArray(REALSXP, dimsVector);
        double* rArray_ptr = REAL(rArray);
        for (int i=0; i<len; i++)
          rArray_ptr[i] = pData[i];
        if(og_typenum == NPY_DATETIME) {
          for (int i = 0; i<len; i++)
            rArray_ptr[i] /= 1e9; // ns to s
          RObject klass_(Rf_allocVector(STRSXP, 2));
          SEXP klass = klass_;
          Rf_setAttrib(rArray, R_ClassSymbol, klass); // protects, but rchk doesn't know
          SET_STRING_ELT(klass, 0, Rf_mkChar("POSIXct"));
          SET_STRING_ELT(klass, 1, Rf_mkChar("POSIXt"));
        }
        break;
      }

      case NPY_CDOUBLE: {
        npy_complex128* pData = (npy_complex128*)PyArray_DATA(array);
        rArray = Rf_allocArray(CPLXSXP, dimsVector);
        Rcomplex* rArray_ptr = COMPLEX(rArray);
        for (int i=0; i<len; i++) {
          npy_complex128 data = pData[i];
          Rcomplex cpx;
          cpx.r = data.real;
          cpx.i = data.imag;
          rArray_ptr[i] = cpx;
        }
        break;
      }

      case NPY_STRING:
      case NPY_VSTRING:
      case NPY_UNICODE: {

        rArray = Rf_allocArray(STRSXP, dimsVector);
        RObject protectArray(rArray);

        // special case 0-size vectors, because np.nditer() throws:
        // ValueError: Iteration of zero-sized operands is not enabled
        if (Rf_length(rArray) == 0)
          break;

        static PyObject* nditerArgs = []() {
          PyObject* flags = PyTuple_New(1);
          // iterating over a StringDType requires us to pass 'refs_ok' flag,
          // since StringDTypes are really refs under the hood.
          PyTuple_SetItem(flags, 0, as_python_str("refs_ok")); // steals ref
          PyObject* args = PyTuple_New(2);
          PyTuple_SetItem(args, 1, flags); // steals ref
          return args;
        }();
        // PyTuple_SetItem steals reference the array, but it's already wrapped
        // into PyObjectPtr earlier (so it gets deleted after the scope of this function)
        // To avoid trying to delete it twice, we need to increase its ref count here.
        PyTuple_SetItem(nditerArgs, 0, (PyObject*)array);
        Py_IncRef((PyObject*)array);

        PyObjectPtr iter(PyObject_Call(get_np_nditer(), nditerArgs, NULL));
        PyTuple_SetItem(nditerArgs, 0, NULL); // clear ref to array

        if (iter.is_null()) {
          throw PythonException(py_fetch_error());
        }

        for (int i=0; i<len; i++) {
          PyObjectPtr el(PyIter_Next(iter)); // returns an scalar array.
          PyObjectPtr pyStr(PyObject_CallMethod(el, "item", NULL));
          if (pyStr.is_null()) {
            throw PythonException(py_fetch_error());
          }
          set_string_element(rArray, i, pyStr);
        }
        break;
      }

      case NPY_OBJECT: {

        // get python objects
        PyObject** pData = (PyObject**)PyArray_DATA(array);

        // check for all strings
        bool allStrings = true;
        for (npy_intp i=0; i<len; i++) {
          auto el = pData[i];
          if (!is_python_str(el) && !is_pandas_na_like(el)) {
            allStrings = false;
            break;
          }
        }

        // return a character vector if it's all strings
        if (allStrings) {
          rArray = Rf_allocArray(STRSXP, dimsVector);
          RObject protectArray(rArray);
          for (npy_intp i = 0; i < len; i++)
            set_string_element(rArray, i, pData[i]);
          break;
        }

        // otherwise return a list of objects
        rArray = Rf_allocArray(VECSXP, dimsVector);
        RObject protectArray(rArray);
        for (npy_intp i = 0; i < len; i++) {
          SEXP data = py_to_r(pData[i], convert);
          SET_VECTOR_ELT(rArray, i, data);
        }

        break;
      }

    case NPY_VOID: {
      // convert to raw, but only if itemsize is 1.
      // We can probably treat itemsize>1 implicitly as one of the
      // dimensions, but we just don't support it.
      // figureing out how to make that not surprising w.r.t. Fortran or C
      // ordering is non-trivial.

      if (PyArray_ITEMSIZE(array) != 1) {
        simple = false;
        goto cant_convert;
      }
      // ?? do we need to check allignment or endianness?

      npy_ubyte* pData = (npy_ubyte*)PyArray_DATA(array);
      rArray = Rf_allocArray(RAWSXP, dimsVector);
      Rbyte* rArray_ptr = RAW(rArray);
      memcpy(rArray_ptr, pData, len * sizeof(npy_ubyte));
    }
    }

    // return the R Array
    return rArray;

  }

  // check for numpy scalar
  if (isPyArrayScalar(x)) {

    // determine the type to convert to
    PyArray_DescrPtr descrPtr(PyArray_DescrFromScalar(x));
    int og_typenum = descrPtr.get()->type_num;
    int typenum = narrow_array_typenum(og_typenum);
    if (typenum == -1) {
      simple = false;
      goto cant_convert;
    }

    PyObjectPtr x_;
    if(og_typenum == NPY_DATETIME) {
      PyObjectPtr dtype_str(as_python_str("datetime64[ns]"));
      x = PyObject_CallMethod(x, "astype", "O", (PyObject*) dtype_str.get()); // new ref
      x_.assign(x); // decref new array on scope exit
    }

    PyArray_DescrPtr toDescr(PyArray_DescrFromType(typenum));

    // convert to R type (guaranteed to by NPY_BOOL, NPY_LONG, or NPY_DOUBLE
    // as per the contract of narrow_arrow_typenum)
    switch(typenum) {

    case NPY_BOOL:
    {
      npy_bool value;
      PyArray_CastScalarToCtype(x, (void*)&value, toDescr);
      return LogicalVector::create(value);
    }

    case NPY_LONG:
    {
      npy_long value;
      PyArray_CastScalarToCtype(x, (void*)&value, toDescr);
      return IntegerVector::create(value);
    }

    case NPY_DOUBLE:
    {
      npy_double value;
      PyArray_CastScalarToCtype(x, (void*)&value, toDescr);
      SEXP out = PROTECT(Rf_ScalarReal(value)); //  NumericVector::create(value);
      if (og_typenum == NPY_DATETIME) {
        REAL(out)[0] /= 1e9;
        SEXP klass = Rf_allocVector(STRSXP, 2);
        Rf_setAttrib(out, R_ClassSymbol, klass); // protects
        SET_STRING_ELT(klass, 0, Rf_mkChar("POSIXct"));
        SET_STRING_ELT(klass, 1, Rf_mkChar("POSIXt"));
      }
      UNPROTECT(1);
      return out;
    }

    case NPY_CDOUBLE:
    {
      npy_complex128 value;
      PyArray_CastScalarToCtype(x, (void*)&value, toDescr);
      Rcomplex cpx;
      cpx.r = value.real;
      cpx.i = value.imag;
      return ComplexVector::create(cpx);
    }

    default:
    {
      stop("Unsupported array conversion from %d", typenum);
    }

    }

  }

  // bytearray
  if (PyByteArray_Check(x)) {

    auto size = PyByteArray_Size(x);
    if (size == 0)
      return RawVector();

    char* data = PyByteArray_AsString(x);
    return RawVector(data, data + size);

  }


  // r object capsule
  if (is_r_object_capsule(x)) {
    return py_capsule_read(x);
  }


  // default is to return opaque wrapper to python object. we pass convert = true
  // because if we hit this code then conversion has been either implicitly
  // or explicitly requested.

  // mark that the object is not a simple python object
  // i.e., not fast-path convertable by r_to_py_cpp()
  // (this lets us skip checking next time, if we go through this code path twice,
  // once in r_to_py() before dispatch, and again if we invoke r_to_py.default()
  simple = false;

  } // end convert == true && simple == true

  cant_convert:
  Py_IncRef(x);
  return PyObjectRef(x, convert, simple);

}

/* stretchy list, modified from R sources
   CAR of the list points to the last cons-cell
   CDR points to the first.
*/

SEXP NewList(void) {
  SEXP s = Rf_cons(R_NilValue, R_NilValue);
  SETCAR(s, s);
  return s;
}

/* Add named element to the end of a stretchy list */
void GrowList(SEXP args_list, SEXP tag, SEXP dflt) {
  PROTECT(dflt);
  SEXP tmp = PROTECT(Rf_cons(dflt, R_NilValue));
  SET_TAG(tmp, tag);

  SETCDR(CAR(args_list), tmp); // set cdr on the last cons-cell
  SETCAR(args_list, tmp);      // update pointer to last cons cell
  UNPROTECT(2);
}

// [[Rcpp::export]]
SEXP py_get_formals(PyObjectRef callable)
{
  GILScope _gil;
  PyObject* callable_ = callable.get();

  static PyObject *inspect_module = NULL;
  static PyObject *inspect_signature = NULL;
  static PyObject *inspect_Parameter = NULL;
  static PyObject *inspect_Parameter_VAR_KEYWORD = NULL;
  static PyObject *inspect_Parameter_VAR_POSITIONAL = NULL;
  static PyObject *inspect_Parameter_KEYWORD_ONLY = NULL;
  static PyObject *inspect_Parameter_empty = NULL;

  if (!inspect_Parameter_empty)
  {
    // initialize static variables to avoid repeat lookups

    inspect_module = PyImport_ImportModule("inspect");
    if (!inspect_module) throw PythonException(py_fetch_error());

    inspect_signature = PyObject_GetAttrString(inspect_module, "signature");
    if (!inspect_signature) throw PythonException(py_fetch_error());

    inspect_Parameter = PyObject_GetAttrString(inspect_module, "Parameter");
    if (!inspect_Parameter) throw PythonException(py_fetch_error());

    inspect_Parameter_VAR_KEYWORD = PyObject_GetAttrString(inspect_Parameter, "VAR_KEYWORD");
    if (!inspect_Parameter_VAR_KEYWORD) throw PythonException(py_fetch_error());

    inspect_Parameter_VAR_POSITIONAL = PyObject_GetAttrString(inspect_Parameter, "VAR_POSITIONAL");
    if (!inspect_Parameter_VAR_POSITIONAL) throw PythonException(py_fetch_error());

    inspect_Parameter_KEYWORD_ONLY = PyObject_GetAttrString(inspect_Parameter, "KEYWORD_ONLY");
    if (!inspect_Parameter_KEYWORD_ONLY) throw PythonException(py_fetch_error());

    inspect_Parameter_empty = PyObject_GetAttrString(inspect_Parameter, "empty");
    if (!inspect_Parameter_empty) throw PythonException(py_fetch_error());

  }

  PyObjectPtr sig(PyObject_CallFunctionObjArgs(inspect_signature, callable_, NULL));
  if (sig.is_null())
  {
    // inspect.signature() can error on builtins in cpython,
    // or python functions built in C from modules
    // fallback to returning formals of `...`.
    PyErr_Clear();
    SEXP out = PROTECT(Rf_cons(R_MissingArg, R_NilValue));
    SET_TAG(out, R_DotsSymbol);
    UNPROTECT(1);
    return out;
  }

  PyObjectPtr parameters(PyObject_GetAttrString(sig, "parameters"));
  if (parameters.is_null()) throw PythonException(py_fetch_error());

  PyObjectPtr items_method(PyObject_GetAttrString(parameters, "items"));
  if (items_method.is_null()) throw PythonException(py_fetch_error());

  PyObjectPtr parameters_items(PyObject_CallFunctionObjArgs(items_method, NULL));
  if (parameters_items.is_null()) throw PythonException(py_fetch_error());

  PyObjectPtr parameters_iterator(PyObject_GetIter(parameters_items));
  if (parameters_iterator.is_null()) throw PythonException(py_fetch_error());

  RObject r_args(NewList());
  PyObject *item;
  bool has_dots = false;

  while ((item = PyIter_Next(parameters_iterator))) // new ref
  {
    PyObjectPtr item_(item); // auto-decref
    PyObject *name = PyTuple_GetItem(item, 0);  // borrowed reference
    PyObject *param = PyTuple_GetItem(item, 1); // borrowed reference

    PyObjectPtr kind_(PyObject_GetAttrString(param, "kind")); // new ref
    if (kind_.is_null()) throw PythonException(py_fetch_error());
    PyObject *kind = kind_.get();

    if (kind == inspect_Parameter_VAR_KEYWORD ||
        kind == inspect_Parameter_VAR_POSITIONAL)
    {
      if (!has_dots)
      {
          GrowList(r_args, R_DotsSymbol, R_MissingArg);
          has_dots = true;
      }
      continue;
    }

    if (!has_dots && kind == inspect_Parameter_KEYWORD_ONLY)
    {
      GrowList(r_args, R_DotsSymbol, R_MissingArg);
      has_dots = true;
    }

    const char *name_char = PyUnicode_AsUTF8(name);
    if (name_char == NULL) throw PythonException(py_fetch_error());

    SEXP name_sym = Rf_installChar(Rf_mkCharCE(name_char, CE_UTF8));

    SEXP arg_default = R_MissingArg;
    PyObjectPtr param_default(PyObject_GetAttrString(param, "default")); // new ref
    if (param_default.is_null())
      throw PythonException(py_fetch_error());

    if (param_default.get() != inspect_Parameter_empty)
      arg_default = py_to_r(param_default, true); // needs protection before next potential R allocation

    GrowList(r_args, name_sym, arg_default); // protects arg_default
  }

  if (PyErr_Occurred())
    throw PythonException(py_fetch_error());

  return CDR(r_args);
}

bool is_convertible_to_numpy(RObject x) {

  if (!haveNumPy())
    return false;

  int type = TYPEOF(x);

  return
    type == INTSXP  ||
    type == REALSXP ||
    type == LGLSXP  ||
    type == CPLXSXP ||
    type == STRSXP;
}

PyObject* r_to_py_numpy(RObject x, bool convert) {

  int type = x.sexp_type();
  SEXP sexp = x.get__();

  // figure out dimensions for resulting array
  SEXP dim_sexp = Rf_getAttrib(sexp, R_DimSymbol);
  IntegerVector dimensions = (dim_sexp != R_NilValue)
    ? IntegerVector(dim_sexp)
    : IntegerVector::create(Rf_xlength(x));

  int nd = dimensions.length();
  std::vector<npy_intp> dims(nd);
  for (int i = 0; i < nd; i++)
    dims[i] = dimensions[i];

  // get pointer + type for underlying data
  int typenum;
  void* data;
  if (type == INTSXP) {
    if (sizeof(long) == 4)
      typenum = NPY_LONG;
    else
      typenum = NPY_INT;
    data = &(INTEGER(sexp)[0]);
  } else if (type == REALSXP) {
    typenum = NPY_DOUBLE;
    data = &(REAL(sexp)[0]);
  } else if (type == LGLSXP) {
    typenum = NPY_BOOL;
    data = &(LOGICAL(sexp)[0]);
  } else if (type == CPLXSXP) {
    typenum = NPY_CDOUBLE;
    data = &(COMPLEX(sexp)[0]);
  } else if (type == STRSXP) {
    typenum = NPY_OBJECT;
    data = NULL;
  } else if (type == RAWSXP) {
    // NPY_UBYTE (np.uint8) might be a more natural choice,
    // but it can't roundtrip.
    typenum = NPY_VOID;
    data = &(RAW(sexp)[0]);
  } else {
    stop("Matrix type cannot be converted to python (only integer, "
           "numeric, complex, logical, and character matrixes can be "
           "converted");
  }

  int flags = NPY_ARRAY_FARRAY_RO;

  // because R logical vectors are just ints under the
  // hood, we create a strided view of the ints, to avoid
  // an allocation.
  npy_intp* strides = NULL;
  if (typenum == NPY_BOOL) {
    // Hack to allocate some memory
    SEXP strides_s = PROTECT(Rf_allocVector(INTSXP, nd * (sizeof(npy_intp) / sizeof(int))));
    // note: npy_intp is typically 8 bytes, int is 4 bytes. Could hardcode to nd*2 if I
    // had more confidence in npy_intp always being 8 bytes.
    strides = (npy_intp*) INTEGER(strides_s);
    int element_size = sizeof(int);
    for (int i = 0; i < nd; i++) {
      strides[i] = element_size;
      if (dims[i])
        element_size *= dims[i];
    }

  }
  // create the array
  PyObject* array = PyArray_New(&PyArray_Type,
                                nd,
                                &(dims[0]),
                                typenum,
                                strides,
                                data,
                                // itemsize, in bytes. Only consulted if
                                // typenum is unsized (e.g., V, U, S). Otherwise ignored.
                                // RAWSXP is converted to void8 (i.e., V1)
                                typenum == NPY_VOID ? 1 : 0, // itemsize
                                flags,
                                NULL);

  if(typenum == NPY_BOOL)
    UNPROTECT(1); // strides_s

  // check for error
  if (array == NULL)
    throw PythonException(py_fetch_error());

  // if this is a character vector we need to convert and set the elements,
  // otherwise the memory is shared with the underlying R vector
  if (type == STRSXP) {
    void** pData = (void**)PyArray_DATA((PyArrayObject*)array);
    R_xlen_t len = Rf_xlength(x);
    for (R_xlen_t i = 0; i<len; i++) {
      PyObject* pyStr = as_python_str(STRING_ELT(x, i), /*handle_na=*/true);
      pData[i] = pyStr;
    }

  } else {
    // wrap the R object in a capsule that's tied to the lifetime of the matrix
    // (so the R doesn't deallocate the memory while python is still pointing to it)
    PyObjectPtr capsule(py_capsule_new(x));

    // set base object using correct version of the API (detach since this
    // effectively steals a reference to the provided base object)
    if (PyArray_GetNDArrayCFeatureVersion() >= NPY_1_7_API_VERSION) {
      int res = PyArray_SetBaseObject((PyArrayObject *)array, capsule.detach());
      if (res != 0)
        throw PythonException(py_fetch_error());
    } else {
      PyArray_BASE(array) = capsule.detach();
    }
  }

  // return it
  return array;

}

PyObject* r_to_py_cpp(RObject x, bool convert);

// returns a new reference
PyObject* r_to_py(RObject x, bool convert) {
  // if the object bit is not set, we can skip R dispatch
  if (OBJECT(x) == 0)
    return r_to_py_cpp(x, convert);

  if(is_py_object(x)) {
    PyObject* obj = PyObjectRef(x, false).get();
    Py_IncRef(obj);
    return(obj);
  }

  // call the R version and hold the return value in a PyObjectRef (SEXP wrapper)
  // this object will be released when the function returns
  // ideally this would call in userenv, so UseMethod() finds methods there.
  // rchk complains that PyObjectRef(eval()) leaves eval() result unprotected
  // while PyObjectRef constructor might allocate, which it only if throwing
  // an exception... do the indirect construction to avoid the warning.
  RObject ref_sexp(eval_call(r_func_r_to_py, x, Rf_ScalarLogical(convert)));
  PyObjectRef ref(ref_sexp);

  // get the underlying Python object and call Py_IncRef before returning it
  // this allows this function to provide the same memory semantics as the
  // previous C++ version of r_to_py (which is now r_to_py_cpp), which always
  // calls Py_IncRef on Python objects before returning them
  PyObject* obj = ref.get();
  Py_IncRef(obj);

  // return the Python object
  return obj;
}


// convert an R object to a python object (the returned object
// will have an active reference count on it)
// the convert arg is only applicable to R functions that will being wrapped in python functions.
PyObject* r_to_py_cpp(RObject x, bool convert) {
  GILScope _gil;

  int type = x.sexp_type();
  SEXP sexp = x.get__();

  // NULL becomes python None
  // (Py_IncRef since PyTuple_SetItem will steal the passed reference)
  if (x.isNULL()) {
    Py_IncRef(Py_None);
    return Py_None;
  }

  // pass python objects straight through (Py_IncRef since returning this
  // creates a new reference from the caller)
  // This will also initialize python if sexp is an R module proxy
  if (is_py_object(sexp)) {
    PyObject* pyobj = PyObjectRef(sexp).get();
    Py_IncRef(pyobj);
    return pyobj;
    // should we update convert to the new value?
  }

  // convert arrays and matrixes to numpy (throw error if numpy not available)
  if (Rf_getAttrib(sexp, R_DimSymbol) != R_NilValue &&
      requireNumPy()) {
    return r_to_py_numpy(x, convert);
  }

  // integer (pass length 1 vectors as scalars, otherwise pass list)
  if (type == INTSXP) {

    R_xlen_t len = XLENGTH(sexp);
    int* psexp = INTEGER(sexp);

    // handle scalars
    if (len == 1)
      return PyInt_FromLong(psexp[0]);

    PyObjectPtr list(PyList_New(len));
    for (R_xlen_t i = 0; i<len; i++) {
      int value = psexp[i];
      // NOTE: reference to added value is "stolen" by the list
      if (PyList_SetItem(list, i, PyInt_FromLong(value)))
        throw PythonException(py_fetch_error());
    }

    return list.detach();

  }

  // numeric (pass length 1 vectors as scalars, otherwise pass list)
  if (type == REALSXP) {

    // handle scalars
    R_xlen_t len = XLENGTH(sexp);
    double* psexp = REAL(sexp);
    if (len == 1)
      return PyFloat_FromDouble(psexp[0]);

    PyObjectPtr list(PyList_New(len));
    for (R_xlen_t i = 0; i<len; i++) {
      double value = psexp[i];
      // NOTE: reference to added value is "stolen" by the list
      if (PyList_SetItem(list, i, PyFloat_FromDouble(value)))
        throw PythonException(py_fetch_error());
    }

    return list.detach();

  }

  // complex (pass length 1 vectors as scalars, otherwise pass list)
  if (type == CPLXSXP) {

    R_xlen_t len = XLENGTH(sexp);
    Rcomplex* psexp = COMPLEX(sexp);
    if (len == 1) {
      Rcomplex cplx = psexp[0];
      return PyComplex_FromDoubles(cplx.r, cplx.i);
    }

    PyObjectPtr list(PyList_New(len));
    for (R_xlen_t i = 0; i<len; i++) {
      Rcomplex cplx = psexp[i];
      // NOTE: reference to added value is "stolen" by the list
      if (PyList_SetItem(list, i, PyComplex_FromDoubles(cplx.r, cplx.i)))
        throw PythonException(py_fetch_error());
    }

    return list.detach();

  }

  // logical (pass length 1 vectors as scalars, otherwise pass list)
  if (type == LGLSXP) {

    R_xlen_t len = XLENGTH(sexp);
    int* psexp = LOGICAL(sexp);

    if (len == 1)
      return PyBool_FromLong(psexp[0]);

    PyObjectPtr list(PyList_New(len));
    for (R_xlen_t i = 0; i<len; i++) {
      int value = psexp[i];
      // NOTE: reference to added value is "stolen" by the list
      if (PyList_SetItem(list, i, PyBool_FromLong(value)))
        throw PythonException(py_fetch_error());
    }

    return list.detach();

  }

  // character (pass length 1 vectors as scalars, otherwise pass list)
  if (type == STRSXP) {

    R_xlen_t len = XLENGTH(sexp);

    if (len == 1)
      return as_python_str(STRING_ELT(sexp, 0));

    PyObjectPtr list(PyList_New(len));
    for (R_xlen_t i = 0; i<len; i++) {
      // NOTE: reference to added value is "stolen" by the list
      int res = PyList_SetItem(list, i, as_python_str(STRING_ELT(sexp, i)));
      if (res != 0)
        throw PythonException(py_fetch_error());
    }

    return list.detach();
  }

  // bytes
  if (type == RAWSXP) {

    Rcpp::RawVector raw(sexp);
    if (raw.size() == 0)
      return PyByteArray_FromStringAndSize(NULL, 0);

    return PyByteArray_FromStringAndSize(
      (const char*) RAW(raw),
      raw.size());

  }

  // list
  if (type == VECSXP) {

    R_xlen_t len = XLENGTH(sexp);

    // create a dict for names
    if (x.hasAttribute("names")) {
      PyObjectPtr dict(PyDict_New());
      CharacterVector names = x.attr("names");
      SEXP namesSEXP = names;
      for (R_xlen_t i = 0; i<len; i++) {
        const char* name = Rf_translateChar(STRING_ELT(namesSEXP, i));
        PyObjectPtr item(r_to_py(VECTOR_ELT(sexp, i), convert));
        if (PyDict_SetItemString(dict, name, item))
          throw PythonException(py_fetch_error());
      }

      return dict.detach();

    }

    // create a list if there are no names
    PyObjectPtr list(PyList_New(len));
    for (R_xlen_t i = 0; i<len; i++) {
      PyObject* item = r_to_py(VECTOR_ELT(sexp, i), convert);
      // NOTE: reference to added value is "stolen" by the list
      int res = PyList_SetItem(list, i, item);
      if (res != 0)
        throw PythonException(py_fetch_error());
    }

    return list.detach();

  }

  if (type == CLOSXP) {

    // create an R object capsule for the R function
    PyObjectPtr capsule(py_capsule_new(x));
    PyCapsule_SetContext(capsule, (void*)convert);

    // check for a py_function_name attribute
    PyObjectPtr pyFunctionName(r_to_py(x.attr("py_function_name"), convert));

    static PyObject* make_python_function = NULL;
    if (make_python_function == NULL) {
      PyObjectPtr module(py_import("rpytools.call"));
      if (module.is_null())
        throw PythonException(py_fetch_error());

      make_python_function =
        PyObject_GetAttrString(module, "make_python_function");

      if (make_python_function == NULL)
        throw PythonException(py_fetch_error());
    }

    // create the python wrapper function
    PyObjectPtr wrapper(
        PyObject_CallFunctionObjArgs(
          make_python_function,
          capsule.get(),
          pyFunctionName.get(),
          NULL));

    if (wrapper.is_null())
      throw PythonException(py_fetch_error());

    // return the wrapper
    return wrapper.detach();

  }

  // default fallback, wrap the r object in a py capsule
  return py_capsule_new(sexp);

}

// [[Rcpp::export]]
PyObjectRef r_to_py_impl(RObject object, bool convert) {
  GILScope _gil;
  return py_ref(r_to_py_cpp(object, convert), convert);
}

class AllowPyThreadsScope
{
private:
  PyThreadState *_save;

public:
  AllowPyThreadsScope() {
    _save = PyEval_SaveThread();
  }

  ~AllowPyThreadsScope() {
    PyEval_RestoreThread(_save);
  }
};

// custom module used for calling R functions from python wrappers


void safe_print_value(SEXP value, const char* warn_print_failed) {
    // Use R_ToplevelExec to safely execute Rf_PrintValue with an inlined lambda
    if (!R_ToplevelExec([](void* data) -> void {
            Rf_PrintValue(static_cast<SEXP>(data));
        }, value)) {
        // Handle the case where Rf_PrintValue failed due to an error
        Rf_warning("%s", warn_print_failed);
    }
}

// must be called from the main thread, accesses the R api
// returns length 2 tuple: (<return-value>, None), or (None, <signaled-error>)
struct PythonCallResult {
  PyObject* value;
  PyObject* exception;
};

PythonCallResult actually_call_r_function(PyObject* args, PyObject* keywords) {
  GILScope _gil;
  PyObject* capsule = PyTuple_GetItem(args, 0);
  RObject rFunction = py_capsule_read(capsule);

  bool convert = (bool)PyCapsule_GetContext(capsule);

  // convert remainder of positional arguments to R list
  PyObjectPtr funcArgs(PyTuple_GetSlice(args, 1, PyTuple_Size(args)));
  List rArgs;
  if (convert) {
    rArgs = py_to_r(funcArgs, convert);
  } else {
    Py_ssize_t len = PyTuple_Size(funcArgs);
    std::vector<PyObjectRef> values;
    values.reserve(len);
    for (Py_ssize_t index = 0; index<len; index++) {
      PyObject* item = PyTuple_GetItem(funcArgs, index); // borrowed
      Py_IncRef(item);
      values.push_back(py_ref(item, convert));
    }
    rArgs = List(values.begin(), values.end());
  }

  // get keyword arguments
  List rKeywords;
  if (keywords != NULL) {

    if (convert) {
      rKeywords = py_to_r(keywords, convert);
    } else {

      PyObject *key, *value;
      Py_ssize_t pos = 0;
      std::vector<PyObjectRef> values;
      std::vector<std::string> names;

      Py_ssize_t size = PyDict_Size(keywords);
      names.reserve(size);
      values.reserve(size);

      // NOTE: PyDict_Next uses borrowed references,
      // so anything we return should be Py_IncRef'd
      while (PyDict_Next(keywords, &pos, &key, &value)) {
        PyObjectPtr str(PyObject_Str(key));
        names.push_back(as_std_string(str));

        Py_IncRef(value);
        values.push_back(py_ref(value, convert));
      }

      rKeywords = List(values.begin(), values.end());
      rKeywords.names() = Rcpp::wrap(names);
    }

  }

  static SEXP call_r_function_s = []() {
    // Use an expression that deparses nicely for traceback printing purposes
    SEXP cl = Rf_lang3(Rf_install(":::"), Rf_install("reticulate"), Rf_install("call_r_function"));
    R_PreserveObject(cl);
    return cl;
  }();

  RObject call_r_func_call(Rf_lang4(call_r_function_s, rFunction, rArgs, rKeywords));

  // PythonCallResult res = {NULL, NULL};
  PyObject* exception = NULL;
  SEXP err_cnd = NULL;

  try {
    // use current_env() here so that in case of error, rlang::trace_back()
    // prints this frame as a node of the parent rather than a top-level call.
    // Rf_eval() is safe to use here because call_r_function() setups up calling handlers
    // so there should be no risk of a longjump from here
    // Technically we don't need to protct env, since
    // it would already be protected by it's inclusion in the R callstack frames,
    // but rchk flags it anyway, and so ...
    RObject env(current_env());
    Rcpp::List result;
    {
      AllowPyThreadsScope _allow_threads;
      result = Rf_eval(call_r_func_call, env);
    }

    // result is either
    // (return_value, NULL) or
    // (NULL, condition)
    if (result[1] == R_NilValue) { // no error
      PyObject* value = r_to_py(result[0], convert);
      return PythonCallResult {value, NULL};
    }


    // R signaled an error
    // Convert the R error condition to a Python Exception
    err_cnd = result[1];
    exception = r_to_py(err_cnd, true);

  } catch(const Rcpp::internal::InterruptedException& e) {
    // should rarely happen, since we also set an R level interrupt handler
    exception = PyExc_KeyboardInterrupt;
  } catch(const std::exception& e) {
    exception = PyUnicode_FromString(e.what());
  } catch(...) {
    exception = PyUnicode_FromString("(Unknown exception occurred)");
  }

  if (exception == NULL) {
    REprintf("Exception raised when converting R error to Python Exception.");
    if (PyErr_Occurred()) PyErr_Print();
    safe_print_value(err_cnd, "Printing the R error condition raised an error");
  }

  return PythonCallResult{NULL, exception};
}

extern "C" PyObject* call_r_function(PyObject *self, PyObject* args, PyObject* keywords)
{

  GILScope _gil;
  PythonCallResult res;
  if(is_main_thread()) {
    res = actually_call_r_function(args, keywords);
  } else {
    static PyObject* safe_call_r_function_on_main_thread = []() -> PyObject* {
      PyObjectPtr thread_tools(PyImport_ImportModule("rpytools.thread"));
      return PyObject_GetAttrString(thread_tools, "safe_call_r_function_on_main_thread");
    }();

    PyObjectPtr res_(PyObject_Call(safe_call_r_function_on_main_thread, args, keywords));

    PyObject *exception = PyTuple_GetItem(res_, 1);
    if (exception == Py_None) { // No Exception raised

      PyObject *value = PyTuple_GetItem(res_, 0);
      Py_IncRef(value);
      res = PythonCallResult{value, NULL};

    } else { // Exception raised

      Py_IncRef(exception);
      res = PythonCallResult {NULL, exception};

    }
  }

  if (res.value)
    return res.value;

  // Prepare to raise the Exception
  // The Python API requires that we separately provide the exception type
  PyObject* exc = res.exception;
  PyObject* exc_type;

  // If the condition converted to an Exception instance,
  // Take the type from that. This is the most common path.
  if (PyExceptionInstance_Check(exc)) {
    exc_type = (PyObject *)Py_TYPE(exc);
  }

  // Also accept a string for simplicity.
  else if (PyUnicode_Check(exc)) {

    if (PyUnicode_CompareWithASCIIString(exc, "KeyboardInterrupt") == 0) {
      exc_type = PyExc_KeyboardInterrupt;
      Py_DecRef(exc);
      exc = NULL;
    } else {
      exc_type = PyExc_RuntimeError;
    }

  }
  // the following two cases should never happen, but catch them just in case
  // The calling handler returned a BaseException class, (not an instance of an Exception)
  else if (PyExceptionClass_Check(exc)) {
    exc_type = exc;
    exc = NULL;
  }
  // Catch-all fallback
  else {
    exc_type = PyExc_RuntimeError;
    if (exc == NULL) {
      exc = PyUnicode_FromString("(Unknown exception)");
    }
  }

  // Raise the exception
  PyErr_SetObject(exc_type, exc);

  // Tell Python we raised an exception
  return NULL;
}

struct PythonCall {
  PythonCall(PyObject* func, PyObject* data) : func(func), data(data) {
    Py_IncRef(func);
    Py_IncRef(data);
  }
  ~PythonCall() {
    Py_DecRef(func);
    Py_DecRef(data);
  }
  PyObject* func;
  PyObject* data;
private:
  PythonCall(const PythonCall& other);
  PythonCall& operator=(const PythonCall&);
};

int call_python_function(void* data) {

  // cast to call
  PythonCall* call = (PythonCall*)data;

  // call the function
  PyObject* arg = py_is_none(call->data) ? NULL : call->data;
  PyObjectPtr res(PyObject_CallFunctionObjArgs(call->func, arg, NULL));

  // delete the call object (will decref the members)
  delete call;

  // return status as per https://docs.python.org/3/c-api/init.html#c.Py_AddPendingCall
  if (!res.is_null())
    return 0;
  else
    return -1;
}


extern "C" PyObject* schedule_python_function_on_main_thread(
                PyObject *self, PyObject* args, PyObject* keywords) {


  // arguments are the python function to call and an optional data argument
  // capture them and then incref them so they survive past this call (we'll
  // decref them in the call_python_function callback)
  PyObject* func = PyTuple_GetItem(args, 0);
  PyObject* data = PyTuple_GetItem(args, 1);

  // create the call object (the func and data will be automaticlaly incref'd then
  // decrefed when the call object is destroyed)
  PythonCall* call = new PythonCall(func, data);

  // Schedule calling the function. Note that we have at least one report of Py_AddPendingCall
  // returning -1, the source code for Py_AddPendingCall is here:
  // https://github.com/python/cpython/blob/faa135acbfcd55f79fb97f7525c8aa6f5a5b6a22/Python/ceval.c#L321-L361
  // From this it looks like it can fail if:
  //
  //   (a) It can't acquire the _PyRuntime.ceval.pending.lock after 100 tries; or
  //   (b) There are more than NPENDINGCALLS already queued
  //
  // As a result we need to check for failure and then sleep and retry in that case.
  //
  // This could in theory result in waiting "forever" but note that if we never successfully
  // add the pending call then we will wait forever anyway as the result queue will
  // never be signaled, i.e. see this code which waits on the call:
  // https://github.com/rstudio/reticulate/blob/b507f954dc08c16710f0fb39328b9770175567c0/inst/python/rpytools/generator.py#L27-L36)
  //
  // As a diagnostic for perverse failure to schedule the call, print a message to stderr
  // every 60 seconds
  //
  const size_t wait_ms = 100;
  size_t waited_ms = 0;
  while(true) {

    // try to schedule the pending call (exit loop on success)
    if (Py_AddPendingCall(call_python_function, call) == 0)
      break;

    // otherwise sleep for wait_ms

    tthread::this_thread::sleep_for(tthread::chrono::milliseconds(wait_ms));

    // increment total wait time and print a warning every 60 seconds
    waited_ms += wait_ms;
    if ((waited_ms % 60000) == 0)
      PySys_WriteStderr("Waiting to schedule call on main R interpeter thread...\n");
  }

  pending_py_calls_notifier::notify();

  // return none
  Py_IncRef(Py_None);
  return Py_None;
}

static void (*s_interrupt_handler)(int) = nullptr;

static int win32_interrupt_handler(long unsigned int ignored) {
  std::cerr << __func__ << std::endl;
  if (s_interrupt_handler != nullptr) {
    s_interrupt_handler(SIGINT);
  }
  return TRUE;
}

template <typename T>
static PyOS_sighandler_t reticulate_setsig(int signum, T&& handler) {

#ifdef _WIN32
    std::cerr << "Setting up console handler" << std::endl;
    s_interrupt_handler = handler;
    SetConsoleCtrlHandler(NULL, FALSE);
    SetConsoleCtrlHandler(win32_interrupt_handler, FALSE);
    SetConsoleCtrlHandler(win32_interrupt_handler, TRUE);
#endif

  std::cerr << "Setting signal handler" << std::endl;
  return PyOS_setsig(signum, handler);

}


static void interrupt_handler(int signum) {

  // This handler is called by the OS when signaling a SIGINT
  std::cerr << "Hello from interrupt handler" << std::endl;

  // Tell R that an interrupt is pending. This will cause R to signal an
  // "interrupt" R condition next time R_CheckUserInterrupt() is called
  R_interrupts_pending = 1;

  // Tell Python there is an interrupt pending. Internally, this calls Python
  // trip_signal(), but does not raise the exception yet. This will cause Python
  // to call the installed Python handler next time PyCheckSignals() is called.
  // The installed Python handler will then be expected to raise a
  // KeyboardInterrupt exception.
  //
  // This does *not* need the GIL, it is safe to call from the c handler.
  PyErr_SetInterrupt();

  // Now, it is a race between R and Python to handle the interrupt.
  // i.e., if R_CheckUserInterrupt() or PyCheckSignals(), is called first.

  // Reinstall this C handler, as it may have been cleared when invoked by the OS
  reticulate_setsig(signum, interrupt_handler);

}


PyOS_sighandler_t orig_interrupt_handler = NULL;

PyOS_sighandler_t install_interrupt_handlers_() {
  // Installs a C handler and a Python handler
  // Exported as an R symbol in case the user did some action that cleared the
  // handlers, e.g., calling signal.signal() in Python, and wants to restore
  // the correct handler.

  // First, install the Python handler
  GILScope _gil;
  PyObject *main = PyImport_AddModule("__main__"); // borrowed ref
  PyObject *main_dict = PyModule_GetDict(main); // borrowed ref
  PyObjectPtr locals(PyDict_New());

  const char* string =
    "from rpycall import python_interrupt_handler\n"
    "from signal import signal, SIGINT\n"
    "signal(SIGINT, python_interrupt_handler)\n";

  PyObjectPtr result(PyRun_StringFlags(string, Py_file_input, main_dict, locals, NULL));
  if (result.is_null()) {
    PyErr_Print();
    Rcpp::warning("Failed to set interrupt signal handlers");
    return NULL;
  }

  // install the C handler.
  //
  // This *must* be after setting the Python handler, because signal.signal()
  // will also reset the OS C handler to one that is not aware of R.
  return reticulate_setsig(SIGINT, interrupt_handler);
}

// [[Rcpp::export]]
void install_interrupt_handlers() {
  install_interrupt_handlers_();
}

extern "C"
PyObject* python_interrupt_handler(PyObject *module, PyObject *args)
{
  // This handler is called by Python from PyCheckSignals(), if
  // it sees that trip_signals() had been called.

  // args will be (signalnum, frame), but we ignore them
  GILScope _gil;
  if (R_interrupts_pending == 0) {
    // R won the race to handle the interrupt. The interrupt has already been
    // signaled as an R condition. There is nothing for this handler to do.
     Py_IncRef(Py_None); return Py_None;
  }

  if (R_interrupts_suspended) {
    // Can't handle the interrupt right now, reschedule self.
    //
    // Note, if this rescheduling approach ends up being too aggressive, we can
    // alternatively rely on rescheduling by the event polling worker, which
    // already runs on a throttled schedule. (Perhaps deferring to that after `n`
    // of these aggressive reschedules).
    PyErr_SetInterrupt();
    Py_IncRef(Py_None); return Py_None;
  }

  // Tell R we handled the interrupt and raise a KeyboardInterrupt exception.
  R_interrupts_pending = 0;
  PyErr_SetNone(PyExc_KeyboardInterrupt);
  return NULL;
}


PyMethodDef RPYCallMethods[] = {
  { "call_r_function", (PyCFunction)call_r_function,
    METH_VARARGS | METH_KEYWORDS, "Call an R function" },
  { "schedule_python_function_on_main_thread", (PyCFunction)schedule_python_function_on_main_thread,
    METH_VARARGS | METH_KEYWORDS, "Call a Python function on the main thread" },
  { "python_interrupt_handler", (PyCFunction)python_interrupt_handler,
    METH_VARARGS, "Handle an interrupt signal" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef RPYCallModuleDef = {
  PyModuleDef_HEAD_INIT,
  "rpycall",
  NULL,
  -1,
  RPYCallMethods,
  NULL,
  NULL,
  NULL,
  NULL
};

extern "C" PyObject* initializeRPYCall(void) {
  return PyModule_Create(&RPYCallModuleDef, _PYTHON3_ABI_VERSION);
}


// [[Rcpp::export]]
void py_activate_virtualenv(const std::string& script) {
  GILScope _gil;

  // import runpy
  PyObjectPtr runpy_module(PyImport_ImportModule("runpy"));
  if (runpy_module.is_null())
    throw PythonException(py_fetch_error());

  // get ref to runpy.run_path()
  PyObjectPtr run_path_func(PyObject_GetAttrString(runpy_module, "run_path"));
  if (run_path_func.is_null())
    throw PythonException(py_fetch_error());

  // make a Python string of the script path
  PyObjectPtr py_script_path(PyUnicode_FromString(script.c_str()));
  if (py_script_path.is_null())
    throw PythonException(py_fetch_error());

  // Call runpy.run_path(script_path) function
  PyObjectPtr result(PyObject_CallFunctionObjArgs(run_path_func, py_script_path.get(), NULL));
  if (result.is_null())
    throw PythonException(py_fetch_error());

}

void trace_print(int threadId, PyFrameObject *frame) {
  std::string tracemsg = "";
  while (NULL != frame) {
    std::string filename = as_std_string(frame->f_code->co_filename);
    std::string funcname = as_std_string(frame->f_code->co_name);
    tracemsg = funcname + " " + tracemsg;

    frame = frame->f_back;
  }

  tracemsg = "THREAD: [" + tracemsg + "]\n";
  PySys_WriteStderr(tracemsg.c_str());
}

void trace_thread_main(void* aArg) {


  int* tracems = (int*)aArg;

  while (true) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyThreadState* pState = PyGILState_GetThisThreadState();

    while (pState != NULL) {
      trace_print(pState->thread_id, pState->frame);
      pState = PyThreadState_Next(pState);
    }

    PyGILState_Release(gstate);

    tthread::this_thread::sleep_for(tthread::chrono::milliseconds(*tracems));
  }
}

tthread::thread* ptrace_thread;
void trace_thread_init(int tracems) {
  ptrace_thread = new tthread::thread(trace_thread_main, &tracems);
}

namespace {

#ifdef _WIN32

SEXP main_process_python_info_win32() {
  // NYI
  return R_NilValue;
}

#else

// a simplified version of loadSymbol adopted from libpython.cpp
void loadSymbol(void* pLib, const std::string& name, void** ppSymbol) {
  *ppSymbol = NULL;
  *ppSymbol = ::dlsym(pLib, name.c_str());
}

SEXP main_process_python_info_unix() {

  // bail early if we already know that Python symbols are not available
  // (initialize as true to first assume symbols are available)
  static bool py_symbols_available = true;
  if (!py_symbols_available)
    return R_NilValue;

  // attempt to load some required Python symbols
  void* pLib = NULL;
  pLib = ::dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);

  if (Py_IsInitialized == NULL)
    loadSymbol(pLib, "Py_IsInitialized", (void**) &Py_IsInitialized);

  if (Py_GetVersion == NULL)
    loadSymbol(pLib, "Py_GetVersion", (void**) &Py_GetVersion);

  ::dlclose(pLib);

  // check and see if loading of these symbols failed
  if (Py_IsInitialized == NULL || Py_GetVersion == NULL) {
    py_symbols_available = false;
    return R_NilValue;
  }

  // retrieve DLL info
  Dl_info dinfo;
  if (dladdr((void*) Py_IsInitialized, &dinfo) == 0) {
    py_symbols_available = false;
    return R_NilValue;
  }

  if (PyGILState_Release == NULL) {
    loadSymbol(pLib, "PyGILState_Release", (void**)&PyGILState_Release);
    // PyGILState_Ensure is always not NULL, since we set it in reticulate_init()
    loadSymbol(pLib, "PyGILState_Ensure", (void**)&PyGILState_Ensure);
  }

  GILScope scope;

  // read Python program path
  std::string python_path;
  if (Py_GetVersion()[0] >= '3') {
    loadSymbol(pLib, "Py_GetProgramFullPath", (void**) &Py_GetProgramFullPath); // deprecated in 3.13
    const std::wstring wide_python_path(Py_GetProgramFullPath());
    python_path = to_string(wide_python_path);
  } else {
    loadSymbol(pLib, "Py_GetProgramFullPath", (void**) &Py_GetProgramFullPath_v2);
    python_path = Py_GetProgramFullPath_v2();
  }

  RObject libpython;
  // read libpython file path
  if (strcmp(dinfo.dli_fname, python_path.c_str()) == 0 ||
      strcmp(dinfo.dli_fname, "python") == 0) {
    // if the library is the same as the executable, it's probably a PIE.
    // Any consequent dlopen on the PIE may fail, return NA to indicate this.
    // when R is embedded by rpy2, dli_fname can be 'python'
    libpython = Rf_ScalarString(R_NaString);
  } else {
    libpython = Rf_mkString(dinfo.dli_fname);
  }

  return List::create(_["python"] = python_path,
                      _["libpython"] = libpython);

}

#endif

} // end anonymous namespace

// [[Rcpp::export]]
SEXP main_process_python_info() {

#ifdef _WIN32
  return main_process_python_info_win32();
#else
  return main_process_python_info_unix();
#endif

}


// [[Rcpp::export]]
void py_clear_error() {
  GILScope _gil;
  DBG("Clearing Python errors.");
  PyErr_Clear();
}

// [[Rcpp::export]]
void py_initialize(const std::string& python,
                   const std::string& libpython,
                   const std::string& pythonhome,
                   const std::string& virtualenv_activate,
                   int python_major_version,
                   int python_minor_version,
                   bool interactive,
                   const std::string& numpy_load_error) {

  // set python3 and interactive flags
  s_isPython3 = python_major_version == 3;
  s_isInteractive = interactive;

  if(!s_isPython3)
    warning("Python 2 reached EOL on January 1, 2020. Python 2 compatability will be removed in an upcoming reticulate release.");

  // load the library
  std::string err;
  if (!libPython().load(libpython, python_major_version, python_minor_version, &err))
    stop(err);

  if (is_python3()) {

    if (Py_IsInitialized()) {
      // if R is embedded in a python environment, rpycall has to be loaded as a regular
      // module.
      GILScope scope;
      PyImport_AddModule("rpycall");
      PyDict_SetItemString(PyImport_GetModuleDict(), "rpycall", initializeRPYCall());

    } else {

      // set program name
      s_python_v3 = to_wstring(python);
      Py_SetProgramName_v3(const_cast<wchar_t*>(s_python_v3.c_str()));

      // set program home
      s_pythonhome_v3 = to_wstring(pythonhome);
      Py_SetPythonHome_v3(const_cast<wchar_t*>(s_pythonhome_v3.c_str()));

      // add rpycall module
      PyImport_AppendInittab("rpycall", &initializeRPYCall);

      // initialize python
      Py_InitializeEx(0); // 0 means "do not install signal handlers"
      s_was_python_initialized_by_reticulate = true;
      const wchar_t *argv[1] = {s_python_v3.c_str()};
      PySys_SetArgv_v3(1, const_cast<wchar_t**>(argv));

      orig_interrupt_handler = install_interrupt_handlers_();
    }

  } else { // python2

    // set program name
    s_python = python;
    Py_SetProgramName(const_cast<char*>(s_python.c_str()));

    // set program home
    s_pythonhome = pythonhome;
    Py_SetPythonHome(const_cast<char*>(s_pythonhome.c_str()));

    if (!Py_IsInitialized()) {
      // initialize python
      Py_InitializeEx(0);
      s_was_python_initialized_by_reticulate = true;
    }

    // add rpycall module
    Py_InitModule4("rpycall", RPYCallMethods, (char *)NULL, (PyObject *)NULL,
                      _PYTHON_API_VERSION);

    const char *argv[1] = {s_python.c_str()};
    PySys_SetArgv(1, const_cast<char**>(argv));

    orig_interrupt_handler = install_interrupt_handlers_();
    reticulate_setsig(SIGINT, interrupt_handler);
  }

  s_main_thread = tthread::this_thread::get_id();
  s_is_python_initialized = true;
  GILScope _gil;

  // initialize type objects
  initialize_type_objects(is_python3());

  // execute activate_this.py script for virtualenv if necessary
  if (!virtualenv_activate.empty())
    py_activate_virtualenv(virtualenv_activate);

  // resovlve numpy
  if (numpy_load_error.empty())
    import_numpy_api(is_python3(), &s_numpy_load_error);
  else
    s_numpy_load_error = numpy_load_error;

  // initialize trace
  Function sysGetEnv("Sys.getenv");
  RObject tracems_env_( sysGetEnv("RETICULATE_DUMP_STACK_TRACE", 0) );
  std::string tracems_env = as<std::string>(tracems_env_);
  int tracems = ::atoi(tracems_env.c_str());
  if (tracems > 0)
    trace_thread_init(tracems);

  // poll for events while executing python code
  reticulate::event_loop::initialize();

  pending_py_calls_notifier::initialize([]() {
    GILScope _gil;
    R_ToplevelExec([](void* data) {
      // TODO: report back errors to the python thread
      Py_MakePendingCalls();
    }, nullptr);
    flush_std_buffers();
  });
}

bool is_py_finalized = false;

// [[Rcpp::export]]
void py_finalize() {

  if (R_ParseEvalString(".globals$finalized", ns_reticulate) != R_NilValue)
    stop("py_finalize() can only be called once per R session");

  reticulate::event_loop::deinitialize(/*wait =*/ false);
  pending_py_calls_notifier::deinitialize();

  // We shouldn't call PyFinalize() if R is embedded in Python. https://github.com/rpy2/rpy2/issues/872
  if(!s_is_python_initialized || !s_was_python_initialized_by_reticulate)
    return;

  {
    PyGILState_Ensure();
    Py_MakePendingCalls();
    if (orig_interrupt_handler)
      reticulate_setsig(SIGINT, orig_interrupt_handler);
    is_py_finalized = true;
    Py_Finalize();
  }

  {
    // To allow to call py_initialize() again would take:
    // - tracking and invalidating all objects declared in functions with
    //   `static PyObject*` (including static modules like numpy).
    // - unloading the library: libpython::SharedLibrary::unload(&error);
    // - setting all loaded symbols to NULL;
    // - check if internal symbols Python loads persist in the process, and
    //   if they need to be somehow discovered and unloaded.
    // - ... problably other things too.
  }

  s_is_python_initialized = false;
  s_was_python_initialized_by_reticulate = false;

  // Make sure that attempting to get the gil again will call
  // `ensure_python_initialized()`, which will now throw an error.
  R_ParseEvalString("local({ "
      "rm(list = names(.globals), envir = .globals); " // clear R-level references to previous config or python objects
      ".globals$finalized <- TRUE; "
      ".globals$py_repl_active <- FALSE; " // used by IDE?
    "})",
    ns_reticulate);
  PyGILState_Ensure = &_initialize_python_and_PyGILState_Ensure;

  // reticulate::event_loop::deinitialize(/*wait =*/ true);
}

// [[Rcpp::export]]
bool py_is_none(PyObjectRef x) {
  GILScope _gil;
  return py_is_none(x.get());
}

// [[Rcpp::export]]
bool py_compare_impl(PyObjectRef a, PyObjectRef b, const std::string& op) {

  GILScope _gil;
  int opcode;
  if (op == "==")
    opcode = Py_EQ;
  else if (op == "!=")
    opcode = Py_NE;
  else if (op == ">")
    opcode = Py_GT;
  else if (op == ">=")
    opcode = Py_GE;
  else if (op == "<")
    opcode = Py_LT;
  else if (op == "<=")
    opcode = Py_LE;
  else
    stop("Unexpected comparison operation " + op);

  // do the comparison
  int res = PyObject_RichCompareBool(a, b, opcode);
  if (res == -1)
    throw PythonException(py_fetch_error());
  else
    return res == 1;
}

// [[Rcpp::export]]
CharacterVector py_str_impl(PyObjectRef x) {
  GILScope _gil;
  if (!is_python_str(x)) {

    PyObjectPtr str(PyObject_Str(x));
    if (str.is_null())
      throw PythonException(py_fetch_error());

    return CharacterVector::create(as_utf8_r_string(str));

  }

  return CharacterVector::create(as_utf8_r_string(x));

}


//' @export
//' @rdname py_str
// [[Rcpp::export]]
SEXP py_repr(PyObjectRef object) {
  GILScope _gil;

  if(py_is_null_xptr(object))
    return CharacterVector::create(String("<pointer: 0x0>"));

  PyObjectPtr repr(PyObject_Repr(object));

  if (repr.is_null())
    throw PythonException(py_fetch_error());

  return  CharacterVector::create(as_utf8_r_string(repr));
}


// [[Rcpp::export]]
void py_print(PyObjectRef x) {
  CharacterVector out = py_str_impl(x);
  Rf_PrintValue(out);
  Rcout << std::endl;
}

// [[Rcpp::export]]
bool py_is_function(PyObjectRef x) {
  GILScope _gil;
  return PyFunction_Check(x) == 1;
}




// [[Rcpp::export]]
bool py_numpy_available_impl() {
  return haveNumPy();
}


// [[Rcpp::export]]
std::vector<std::string> py_list_attributes_impl(PyObjectRef x) {
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  std::vector<std::string> attributes;
  PyObjectPtr attrs(PyObject_Dir(x_));
  if (attrs.is_null())
    throw PythonException(py_fetch_error());

  Py_ssize_t len = PyList_Size(attrs);
  for (Py_ssize_t index = 0; index<len; index++) {
    PyObject* item = PyList_GetItem(attrs, index);
    attributes.push_back(as_std_string(item));
  }

  return attributes;
}


// [[Rcpp::export]]
SEXP py_get_convert(PyObjectRef x) {
  return Rf_ScalarLogical(x.convert());
}

// [[Rcpp::export]]
SEXP py_set_convert(PyObjectRef x, bool value) {
  Rf_defineVar(sym_convert, Rf_ScalarLogical(value), x.get_refenv());
  return x;
}


// [[Rcpp::export]]
PyObjectRef py_new_ref(PyObjectRef x, SEXP convert) {
  bool convert_ = (convert == R_NilValue)
  ? x.convert() :
  ((bool) Rf_asLogical(convert));

  GILScope _gil;
  PyObject* pyobj = x.get();
  Py_IncRef(pyobj);
  return py_ref(pyobj, convert_);
}


//' Check if a Python object has an attribute
//'
//' Check whether a Python object \code{x} has an attribute
//' \code{name}.
//'
//' @param x A python object.
//' @param name The attribute to be accessed.
//'
//' @return \code{TRUE} if the object has the attribute \code{name}, and
//'   \code{FALSE} otherwise.
//' @export
// [[Rcpp::export]]
bool py_has_attr(PyObjectRef x, const std::string& name) {
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  return py_has_attr(x_, name.c_str());
}

//' Get an attribute of a Python object
//'
//' @param x Python object
//' @param name Attribute name
//' @param silent \code{TRUE} to return \code{NULL} if the attribute
//'  doesn't exist (default is \code{FALSE} which will raise an error)
//'
//' @return Attribute of Python object
//' @export
// [[Rcpp::export]]
PyObjectRef py_get_attr(PyObjectRef x,
                        const std::string& name,
                        bool silent = false)
{
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  PyObject *attr = PyObject_GetAttrString(x_, name.c_str()); // new ref

  if (attr == NULL) {
    if (silent) {
      PyErr_Clear();
      return PyObjectRef(R_NilValue, false);
    } else {
      throw PythonException(py_fetch_error());
    }
  }

  return PyObjectRef(attr, x.convert());
}


//' Set an attribute of a Python object
//'
//' @param x Python object
//' @param name Attribute name
//' @param value Attribute value
//'
//' @export
// [[Rcpp::export(invisible = true)]]
PyObjectRef py_set_attr(PyObjectRef x,
                        const std::string& name,
                        RObject value)
{
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  PyObjectPtr value_(r_to_py(value, x.convert()));
  int res = PyObject_SetAttrString(x_, name.c_str(), value_);
  if (res != 0)
    throw PythonException(py_fetch_error());
  return x;
}

//' Delete an attribute of a Python object
//'
//' @param x A Python object.
//' @param name The attribute name.
//'
//' @export
// [[Rcpp::export(invisible = true)]]
PyObjectRef py_del_attr(PyObjectRef x, const std::string& name)
{
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  int res = PyObject_SetAttrString(x_, name.c_str(), NULL);
  if (res != 0)
    throw PythonException(py_fetch_error());
  return x;
}

//' @rdname py_get_item
//' @export
// [[Rcpp::export]]
PyObjectRef py_get_item(PyObjectRef x, RObject key, bool silent = false)
{
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  PyObjectPtr py_key(r_to_py(key, false));
  PyObject *item = PyObject_GetItem(x_, py_key);
  if (item == NULL) {
    if (silent) {
      PyErr_Clear();
      return PyObjectRef(R_NilValue, false);
    }
    throw PythonException(py_fetch_error());
  }
  return PyObjectRef(item, x.convert());
}

//' @rdname py_get_item
//' @export
// [[Rcpp::export(invisible = true)]]
PyObjectRef py_set_item(PyObjectRef x, RObject key, RObject value)
{
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  PyObjectPtr py_key(r_to_py(key, true));
  PyObjectPtr py_val(r_to_py(value, true));

  int res = PyObject_SetItem(x_, py_key, py_val);
  if (res != 0)
    throw PythonException(py_fetch_error());
  return x;
}

//' @rdname py_get_item
//' @export
// [[Rcpp::export(invisible = true)]]
PyObjectRef py_del_item(PyObjectRef x, RObject key) {
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  PyObjectPtr pyKey(r_to_py(key, true));
  int res = PyObject_DelItem(x_, pyKey.get());
  if (res != 0)
    throw PythonException(py_fetch_error());
  return x;
}


// [[Rcpp::export]]
IntegerVector py_get_attr_types(
    PyObjectRef x,
    const std::vector<std::string>& attrs,
    bool resolve_properties = false)
{
  GILScope _gil;
  PyObject* x_ = x.get(); // ensure python initialized, module proxy resolved
  const int UNKNOWN     =  0;
  const int VECTOR      =  1;
  const int ARRAY       =  2;
  const int LIST        =  4;
  const int ENVIRONMENT =  5;
  const int FUNCTION    =  6;
  PyErrorScopeGuard _g;
  PyObjectPtr type( PyObject_GetAttrString(x_, "__class__") );

  std::size_t n = attrs.size();
  IntegerVector types = no_init(n);
  for (std::size_t i = 0; i < n; i++) {
    const std::string& name = attrs[i];

    // check if this is a property; if so, avoid resolving it unless
    // requested as this could imply running arbitrary Python code
    if (!resolve_properties) {
      PyObjectPtr attr(PyObject_GetAttrString(type, name.c_str()));
      if (attr.is_null())
        PyErr_Clear();
      else if (PyObject_TypeCheck(attr, PyProperty_Type)) {
        types[i] = UNKNOWN;
        continue;
      }
    }

    PyObjectPtr attr(PyObject_GetAttrString(x_, name.c_str()));

    if(attr.is_null()) {
      PyErr_Clear();
      types[i] = UNKNOWN;
    }
    else if (attr.get() == Py_None)
      types[i] = UNKNOWN;
    else if (PyType_Check(attr))
      types[i] = UNKNOWN;
    else if (PyCallable_Check(attr))
      types[i] = FUNCTION;
    else if (PyList_Check(attr)  ||
             PyTuple_Check(attr) ||
             PyDict_Check(attr))
      types[i] = LIST;
    else if (isPyArray(attr))
      types[i] = ARRAY;
    else if (PyBool_Check(attr)   ||
             PyInt_Check(attr)    ||
             PyLong_Check(attr)   ||
             PyFloat_Check(attr)  ||
             is_python_str(attr))
      types[i] = VECTOR;
    else if (PyObject_IsInstance(attr, (PyObject*)PyModule_Type))
      types[i] = ENVIRONMENT;
    else
      // presume that other types are objects
      types[i] = LIST;
  }

  return types;
}


// [[Rcpp::export]]
SEXP py_ref_to_r_with_convert(PyObjectRef x, bool convert) {
  return py_to_r(x, convert);
}

// [[Rcpp::export]]
SEXP py_ref_to_r(PyObjectRef x) {
  return py_ref_to_r_with_convert(x, x.convert());
}




// [[Rcpp::export]]
SEXP py_call_impl(PyObjectRef x, List args = R_NilValue, List keywords = R_NilValue) {

  GILScope _gil;
  bool convert = x.convert();

  // unnamed arguments
  PyObjectPtr pyArgs(PyTuple_New(args.length()));
  if (args.length() > 0) {
    for (R_xlen_t i = 0; i<args.size(); i++) {
      PyObject* arg = r_to_py(args.at(i), convert);
      // NOTE: reference to arg is "stolen" by the tuple
      int res = PyTuple_SetItem(pyArgs, i, arg);
      if (res != 0)
        throw PythonException(py_fetch_error());
    }
  }

  // named arguments
  PyObjectPtr pyKeywords(PyDict_New());
  if (keywords.length() > 0) {
    CharacterVector names = keywords.names();
    SEXP namesSEXP = names;
    for (R_xlen_t i = 0; i<keywords.length(); i++) {
      const char* name = Rf_translateChar(STRING_ELT(namesSEXP, i));
      PyObjectPtr arg(r_to_py(keywords.at(i), convert));
      int res = PyDict_SetItemString(pyKeywords, name, arg);
      if (res != 0)
        throw PythonException(py_fetch_error());
    }
  }

  // call the function
  PyObjectPtr res(PyObject_Call(x, pyArgs, pyKeywords));

  // check for error
  if (res.is_null())
    throw PythonException(py_fetch_error(true));

  // return
  return py_ref(res.detach(), convert);
}

// [[Rcpp::export]]
PyObjectRef py_dict_impl(const List& keys, const List& items, bool convert) {
  GILScope _gil;

  PyObject* dict = PyDict_New();

  for (R_xlen_t i = 0; i < keys.length(); i++) {
    PyObjectPtr key(r_to_py(keys.at(i), convert));
    PyObjectPtr val(r_to_py(items.at(i), convert));
    PyDict_SetItem(dict, key, val);
  }

  return py_ref(dict, convert);

}


// [[Rcpp::export]]
SEXP py_dict_get_item(PyObjectRef dict, RObject key) {
  GILScope _gil;
  PyObject* dict_ = dict.get(); // ensure python initialized, module proxy resolved

  if (!PyDict_CheckExact(dict_)) {
    PyObjectRef ref(py_get_item(dict, key, false));
    if(dict.convert()) {
      // py_get_item_impl returns PyObjectRef always
      // RObject, so that the SEXP is protected while ref destructor is called
      // which rchks flags as a potential issue
      return RObject(py_to_r(ref.get(), true)); // py_to_r() does *not* steal a ref
    } else {
      return ref;
    }
  }

  PyObjectPtr pyKey(r_to_py(key, false));

  // NOTE: returns borrowed reference
  // NOTE: does *not* set an exception if key is missing
  PyObject* item = PyDict_GetItem(dict_, pyKey);
  if (item == NULL)
    item = Py_None;

  return py_to_r(item, dict.convert());
}

// [[Rcpp::export]]
void py_dict_set_item(PyObjectRef dict, RObject key, RObject val) {
  GILScope _gil;
  PyObject* dict_ = dict.get(); // ensure python initialized, module proxy resolved

  if (!PyDict_CheckExact(dict_)) {
    py_set_item(dict, key, val);
    return ;
  }

  PyObjectPtr py_key(r_to_py(key, dict.convert()));
  PyObjectPtr py_val(r_to_py(val, dict.convert()));
  PyDict_SetItem(dict_, py_key, py_val);

}

// [[Rcpp::export]]
int py_dict_length(PyObjectRef dict) {
  GILScope _gil;

  if (!PyDict_CheckExact(dict))
    return PyObject_Size(dict);

  return PyDict_Size(dict);

}

namespace {

PyObject* py_dict_get_keys_impl(PyObject* dict) {

  PyObject* keys = PyDict_Keys(dict);

  if (keys == NULL) {
    PyErr_Clear();
    keys = PyObject_CallMethod(dict, "keys", NULL);
    if (keys == NULL)
      throw PythonException(py_fetch_error());
  }

  return keys;

}

} // end anonymous namespace

// [[Rcpp::export]]
PyObjectRef py_dict_get_keys(PyObjectRef dict) {
  GILScope _gil;
  PyObject* keys = py_dict_get_keys_impl(dict);
  return py_ref(keys, dict.convert());
}

// [[Rcpp::export]]
CharacterVector py_dict_get_keys_as_str(PyObjectRef dict) {
  GILScope _gil;

  // get the dictionary keys
  PyObjectPtr py_keys(py_dict_get_keys_impl(dict));

  // iterate over keys and convert to string
  std::vector<std::string> keys;

  PyObjectPtr it(PyObject_GetIter(py_keys));
  if (it.is_null())
    throw PythonException(py_fetch_error());

  for (PyObject* item = PyIter_Next(it);
       item != NULL;
       item = PyIter_Next(it))
  {
    // decref on scope exit
    PyObjectPtr scope(item);

    // check for python string and use directly
    if (is_python_str(item)) {
      keys.push_back(as_utf8_r_string(item));
      continue;
    }

    // if we don't have a python string, try to create one
    PyObjectPtr str(PyObject_Str(item));
    if (str.is_null())
      throw PythonException(py_fetch_error());

    keys.push_back(as_utf8_r_string(str));

  }

  if (PyErr_Occurred())
    throw PythonException(py_fetch_error());

  return CharacterVector(keys.begin(), keys.end());

}


// [[Rcpp::export]]
PyObjectRef py_tuple(const List& items, bool convert) {
  GILScope _gil;

  R_xlen_t n = items.length();
  PyObject* tuple = PyTuple_New(n);
  for (R_xlen_t i = 0; i < n; i++) {
    PyObject* item = r_to_py(items.at(i), convert);
    // NOTE: reference to arg is "stolen" by the tuple
    int res = PyTuple_SetItem(tuple, i, item);
    if (res != 0)
      throw PythonException(py_fetch_error());
  }

  return py_ref(tuple, convert);

}

// [[Rcpp::export]]
int py_tuple_length(PyObjectRef tuple) {
  GILScope _gil;
  if (!PyTuple_CheckExact(tuple))
    return PyObject_Size(tuple);

  return PyTuple_Size(tuple);

}


// [[Rcpp::export]]
PyObjectRef py_module_import(const std::string& module, bool convert) {
  GILScope _gil;
  PyObject* pModule = py_import(module);
  if (pModule == NULL)
    throw PythonException(py_fetch_error());

  return py_ref(pModule, convert);

}

// [[Rcpp::export]]
void py_module_proxy_import(PyObjectRef proxy) {
  Rcpp::Environment refenv = proxy.get_refenv();
  if (refenv.exists("module")) {
    GILScope _gil;
    Rcpp::RObject r_module = refenv.get("module");
    std::string module = as<std::string>(r_module);
    PyObject* pModule = py_import(module);
    if (pModule == NULL)
      throw PythonException(py_fetch_error());
    proxy.set(pModule);
    refenv.remove("module");
  }// else, if !exists("module", <refenv>),
   // then we're unwinding a recursive py_resolve_module_proxy() call, e.g.:
   // -> py_resolve_module_proxy() -> import() -> py_module_onload() ->
   //  <r-hook-that-forces-a-module-proxy> -> py_resolve_module_proxy()
}



// [[Rcpp::export]]
CharacterVector py_list_submodules(const std::string& module) {

  GILScope _gil;
  std::vector<std::string> modules;

  PyObject* modulesDict = PyImport_GetModuleDict();
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  std::string prefix = module + ".";
  while (PyDict_Next(modulesDict, &pos, &key, &value)) {
    if (PyString_Check(key) && !py_is_none(value)) {
      std::string name = as_std_string(key);
      if (name.find(prefix) == 0) {
        std::string submodule = name.substr(prefix.length());
        if (submodule.find('.') == std::string::npos)
          modules.push_back(submodule);
      }
    }
  }

  return wrap(modules);
}


// [[Rcpp::export]]
SEXP py_run_string_impl(const std::string& code,
                        bool local = false,
                        bool convert = true)
{
  GILScope _gil;
  PyFlushOutputOnScopeExit flush_;
  // retrieve reference to main module dictionary
  // note: both PyImport_AddModule() and PyModule_GetDict()
  // return borrowed references
  PyObject* main = PyImport_AddModule("__main__");
  PyObject* globals = PyModule_GetDict(main);

  if (local) {

    // create dictionary to capture locals
    PyObjectPtr locals(PyDict_New());

    // run the requested code
    PyObjectPtr res(PyRun_StringFlags(code.c_str(), Py_file_input, globals, locals, NULL));
    if (res.is_null())
      throw PythonException(py_fetch_error());

    // return locals dictionary (detach so we don't decref on scope exit)
    return py_ref(locals.detach(), convert);

  } else {

    // run the requested code
    PyObjectPtr res(PyRun_StringFlags(code.c_str(), Py_file_input, globals, globals, NULL));
    if (res.is_null())
      throw PythonException(py_fetch_error());

    // because globals is borrowed, we need to incref here
    Py_IncRef(globals);
    return py_ref(globals, convert);

  }

}

// [[Rcpp::export]]
PyObjectRef py_run_file_impl(const std::string& file,
                      bool local = false,
                      bool convert = true) {
  GILScope _gil;
  FILE* fp = fopen(file.c_str(), "rb");
  if (fp == NULL) stop("Unable to open file '%s'", file);

  PyObject* main = PyImport_AddModule("__main__");  // borrowed reference
  PyObject* globals = PyModule_GetDict(main);       // borrowed reference
  PyObject* locals;

  if (local)
    locals = PyDict_New();  // new reference
  else {
    locals = globals;
    Py_IncRef(locals);
  }

  PyObjectPtr locals_w_finalizer(locals);  // ensure decref on early return

  if (PyDict_SetItemString(locals, "__file__", as_python_str(file)) < 0)
    throw PythonException(py_fetch_error());

  if (PyDict_SetItemString(locals, "__cached__", Py_None) < 0)
    throw PythonException(py_fetch_error());

  PyObjectPtr res(PyRun_FileEx(fp, file.c_str(), Py_file_input, globals,
                               locals, 1));  // 1 here closes fp before it returns

  if (res.is_null())
    throw PythonException(py_fetch_error());

  // try delete dunders; mimic PyRun_SimpleFile() behavior
  if (PyDict_DelItemString(locals, "__file__"))   PyErr_Clear();
  if (PyDict_DelItemString(locals, "__cached__")) PyErr_Clear();

  if (flush_std_buffers() == -1)
    warning(
        "Error encountered when flushing python buffers sys.stderr and "
        "sys.stdout");

  return py_ref(locals_w_finalizer.detach(), convert);
}

// [[Rcpp::export]]
SEXP py_eval_impl(const std::string& code, bool convert = true) {
  GILScope _gil;
  // compile the code
  PyObjectPtr compiledCode;
  if (Py_CompileStringExFlags != NULL)
    compiledCode.assign(Py_CompileStringExFlags(code.c_str(), "reticulate_eval", Py_eval_input, NULL, 0));
  else
    compiledCode.assign(Py_CompileString(code.c_str(), "reticulate_eval", Py_eval_input));


  if (compiledCode.is_null())
    throw PythonException(py_fetch_error());

  // execute the code
  PyObject* main = PyImport_AddModule("__main__");
  PyObject* dict = PyModule_GetDict(main);
  PyObjectPtr local_dict(PyDict_New());
  PyObjectPtr res(PyEval_EvalCode(compiledCode, dict, local_dict));
  if (res.is_null())
    throw PythonException(py_fetch_error());

  // return (convert to R if requested)
  return py_to_r(res, convert);
}

template <int RTYPE>
RObject pandas_nullable_collect_values (PyObject* series) {
  size_t size;
  {
    PyObjectPtr _size(PyObject_GetAttrString(series, "size"));
    if (_size.is_null()) {
      throw PythonException(py_fetch_error());
    }
    size = PyLong_AsLong(_size);
  }

  PyObjectPtr iter(PyObject_GetIter(series));
  if (iter.is_null()) {
    throw PythonException(py_fetch_error());
  }

  Vector<RTYPE> output(size, Rcpp::traits::get_na<RTYPE>());
  for(size_t i=0; i<size; i++) {
    PyObjectPtr item(PyIter_Next(iter));

    if (item.is_null()) {
      throw PythonException(py_fetch_error());
    }

    if (!is_pandas_na(item)) {
      output[i] = Rcpp::as<Vector<RTYPE>>(py_to_r(item, true))[0];
    }
  }

  return output;
}

#define NULLABLE_INTEGERS                                      \
"Int8",                                                        \
"Int16",                                                       \
"Int32",                                                       \
"Int64",                                                       \
"UInt8",                                                       \
"UInt16",                                                      \
"UInt32",                                                      \
"UInt64"


SEXPTYPE nullable_typename_to_sexptype (const std::string& name) {
  const static std::set<std::string> nullable_integers({NULLABLE_INTEGERS});

  if (nullable_integers.find(name) != nullable_integers.end()) {
    return INTSXP;
  } else if (name == "Float32" || name == "Float64") {
    return REALSXP;
  } else if (name == "string") {
    return STRSXP;
  } else if (name == "boolean") {
    return LGLSXP;
  }

  Rcpp::stop("Can't cast column with type name: " + name);
}

// [[Rcpp::export]]
SEXP py_convert_pandas_series(PyObjectRef series_) {
  GILScope _gil;
  PyObject* series = series_.get();

  // extract dtype
  PyObjectPtr dtype(PyObject_GetAttrString(series, "dtype"));
  const auto name = as_std_string(PyObjectPtr(PyObject_GetAttrString(dtype, "name")));

  const static std::set<std::string> nullable_dtypes({
    NULLABLE_INTEGERS,
    "boolean",
    "Float32",
    "Float64",
    "string"
  });

  RObject R_obj;

  // special treatment for pd.Categorical
  if (name == "category") {

    // get actual values and convert to R
    PyObjectPtr cat(PyObject_GetAttrString(series, "cat"));
    PyObjectPtr codes(PyObject_GetAttrString(cat, "codes"));
    PyObjectPtr code_values(PyObject_GetAttrString(codes, "values"));
    RObject R_values = py_to_r(code_values, true);

    // get levels and convert to R
    PyObjectPtr categories(PyObject_GetAttrString(dtype, "categories"));
    PyObjectPtr category_values(PyObject_GetAttrString(categories, "values"));
    RObject R_levels = py_to_r(category_values, true);

    // get "ordered" attribute
    PyObjectPtr ordered(PyObject_GetAttrString(dtype, "ordered"));


    // populate integer vector to hold factor values
    // note that we need to convert 0->1 indexing, and handle NAs
    int* codes_int = INTEGER(R_values);
    int n = Rf_xlength(R_values);

    // values need to start at 1
    IntegerVector factor(n);
    for (int i = 0; i < n; ++i) {
      int code = codes_int[i];
      factor[i] = code == -1 ? NA_INTEGER : code + 1;
    }

    // populate character vector to hold levels
    CharacterVector factor_levels(R_levels);
    factor_levels.attr("dim") = R_NilValue;

    factor.attr("levels") = factor_levels;
    if (PyObject_IsTrue(ordered))
      factor.attr("class") = CharacterVector({"ordered", "factor"});
    else
      factor.attr("class") = "factor";

    R_obj = factor;

  // special treatment for pd.TimeStamp
  // if available, time zone information will be respected,
  // but values returned to R will be in UTC
  } else if (name == "datetime64[ns]" ||

    // if a time zone is present, dtype is "object"
    py_has_attr(series, "dt")) {

    // pd.Series.items() returns an iterator over (index, value) pairs
    PyObjectPtr items(PyObject_CallMethod(series, "items", NULL));

    std::vector<double> posixct;

    while (true) {

      // get next tuple
      PyObjectPtr tuple(PyIter_Next(items));
      if (tuple.is_null()) {
        if (PyErr_Occurred())
          throw PythonException(py_fetch_error());
        else
          break;
      }

     // access value in slot 1
     PyObjectPtr values(PySequence_GetItem(tuple, 1));
     // convert to POSIX timestamp, taking into account time zone (if set)
     PyObjectPtr timestamp(PyObject_CallMethod(values, "timestamp", NULL));

     Datetime R_timestamp;

     // NaT will have thrown "NaTType does not support timestamp"
     if (PyErr_Occurred()) {
       R_timestamp = R_NaN;
       PyErr_Clear();
     } else {
       R_timestamp = py_to_r(timestamp, true);
     }

     posixct.push_back(R_timestamp);

    }

    DatetimeVector R_posixct(posixct.size());
    for (std::size_t i = 0; i < posixct.size(); ++i) {
      R_posixct[i] = posixct[i];
    }

    return R_posixct;


  // Data types starting with Capitalized case are used as the nullable datatypes in
  // Pandas. They use pd.NA to represent missing values and we preserve them in the R
  // arrays.
  } else if (nullable_dtypes.find(name) != nullable_dtypes.end()) {

    // IIFE pattern
    R_obj = [&]() {
      switch (nullable_typename_to_sexptype(name)) {
      case INTSXP: return pandas_nullable_collect_values<INTSXP>(series);
      case REALSXP: return pandas_nullable_collect_values<REALSXP>(series);
      case LGLSXP: return pandas_nullable_collect_values<LGLSXP>(series);
      case STRSXP: return pandas_nullable_collect_values<STRSXP>(series);
      }
      Rcpp::stop("Unsupported data type name: " + name);
    }();

  // default case
  } else {

    PyObjectPtr values(PyObject_GetAttrString(series, "values"));
    R_obj = py_to_r(values, true);

  }

  return R_obj;

}

// [[Rcpp::export]]
SEXP py_convert_pandas_df(PyObjectRef df) {
  GILScope _gil;

  // pd.DataFrame.items() returns an iterator over (column name, Series) pairs
  PyObjectPtr items(PyObject_CallMethod(df, "items", NULL));
  if (!PyIter_Check(items))
    stop("Cannot iterate over object");

  std::vector<RObject> list;

  while (true) {

    // get next tuple
    PyObjectPtr tuple(PyIter_Next(items));
    if (tuple.is_null()) {
      if (PyErr_Occurred())
        throw PythonException(py_fetch_error());
      else
        break;
    }

    // access Series in slot 1
    PyObjectPtr series(PySequence_GetItem(tuple, 1));

    // delegate to py_convert_pandas_series
    PyObjectRef series_ref(series.detach(), true);
    RObject R_obj = py_convert_pandas_series(series_ref);

    list.push_back(R_obj);

  }

  return List(list.begin(), list.end());

}

PyObject* na_mask (SEXP x) {

  const size_t n(XLENGTH(x));
  npy_intp dims(n);

  PyObject* mask(PyArray_SimpleNew(1, &dims, NPY_BOOL));
  if (!mask) throw PythonException(py_fetch_error());

  // Instead of using R's Logical
  // data points to mask 'owned' memory, so we don't need to free it.
  bool* data = (bool*) PyArray_DATA((PyArrayObject*) mask);
  if (!data) throw PythonException(py_fetch_error());

  size_t i;

  // This is modified from R primitive do_isna - backing the `is.na()`:
  // https://github.com/wch/r-source/blob/6b5d4ca5d1e3b4b9e4bbfb8f75577aff396a378a/src/main/coerce.c#L2221
  // Unfortunately couldn't find a simple way to find NA's for whichever atomic type.
  switch (TYPEOF(x)) {
  case LGLSXP:
    for (i = 0; i < n; i++)
      data[i] = (LOGICAL_ELT(x, i) == NA_LOGICAL);
    break;
  case INTSXP:
    for (i = 0; i < n; i++)
      data[i] = (INTEGER_ELT(x, i) == NA_INTEGER);
    break;
  case REALSXP:
    for (i = 0; i < n; i++)
      data[i] = ISNAN(REAL_ELT(x, i));
    break;
  case CPLXSXP:
    for (i = 0; i < n; i++) {
      Rcomplex v = COMPLEX_ELT(x, i);
      data[i] = (ISNAN(v.r) || ISNAN(v.i));
    }
    break;
  case STRSXP:
    for (i = 0; i < n; i++)
      data[i] = (STRING_ELT(x, i) == NA_STRING);
    break;
  }

  return mask;
}

PyObject* r_to_py_pandas_nullable_series (const RObject& column, const bool convert) {

  PyObject* constructor;
  switch (TYPEOF(column)) {
  case INTSXP:
    const static PyObject* IntArray(
        PyObject_GetAttrString(pandas_arrays(), "IntegerArray")
    );
    constructor = const_cast<PyObject*>(IntArray);
    break;
  case REALSXP:
    const static PyObject* FloatArray(
        PyObject_GetAttrString(pandas_arrays(), "FloatingArray")
    );
    constructor =  const_cast<PyObject*>(FloatArray);
    break;
  case LGLSXP:
    const static PyObject* BoolArray(
        PyObject_GetAttrString(pandas_arrays(), "BooleanArray")
    );
    constructor =  const_cast<PyObject*>(BoolArray);
    break;
  case STRSXP:
    const static PyObject* StringArray(
        PyObject_GetAttrString(pandas_arrays(), "StringArray")
    );
    constructor =  const_cast<PyObject*>(StringArray);
    break;
  default:
    Rcpp::stop("R type not handled. Please supply one of int, double, logical or character");
  }

  if (!constructor) {
    // if the constructor is not available it means that the user doesn't have
    // the minimum pandas version.
    // we show a warning and force the numpy construction.
    Rcpp::warning(
      "Nullable data types require pandas version >= 1.2.0. "
      "Forcing numpy cast. Use `options(reticulate.pandas_use_nullable_dtypes = FALSE)` "
      "to disable this warning."
    );

    return r_to_py_numpy(column, convert);
  }

  // strings are not built using np array + mask. Instead they take a
  // np array with OBJECT type, with None's in the place of NA's
  if (TYPEOF(column) == STRSXP) {
    PyObjectPtr args(PyTuple_New(2));
    PyTuple_SetItem(args, 0, (PyObject*)r_to_py_numpy(column, convert));
    PyTuple_SetItem(args, 1, Py_False);

    PyObject* pd_col(PyObject_Call(constructor, args, NULL));

    if (!pd_col) {
      // it's likely that the error is caused by using an old version of pandas
      // that don't accept `None` as a `NA` value.
      // we force the old cast method after a warning.
      Rcpp::warning(
        "String nullable data types require pandas version >= 1.5.0. "
        "Forcing numpy cast. Use `options(reticulate.pandas_use_nullable_dtypes = FALSE)` "
        "to disable this warning."
      );

      return r_to_py_numpy(column, convert);
    }

    return pd_col;
  }

  // tuples own the objects - thus we don't leak the value and mask
  PyObjectPtr args(PyTuple_New(3));
  PyTuple_SetItem(args, 0, (PyObject*)r_to_py_numpy(column, convert)); // value
  PyTuple_SetItem(args, 1, (PyObject*)na_mask(column));                // mask
  PyTuple_SetItem(args, 2, Py_False);                                  // copy=False

  PyObject* pd_col(PyObject_Call(constructor, args, NULL));
  return pd_col;
}

// [[Rcpp::export]]
PyObjectRef r_convert_dataframe(RObject dataframe, bool convert) {
  GILScope _gil;
  Function r_convert_dataframe_column =
    Environment::namespace_env("reticulate")["r_convert_dataframe_column"];

  PyObjectPtr dict(PyDict_New());

  CharacterVector names = dataframe.attr("names");
  // when this is set we cast R atomic vectors to numpy arrays and don't
  // use pandas dtypes that can handle missing values.
  bool nullable_dtypes = option_is_true("reticulate.pandas_use_nullable_dtypes");

  for (R_xlen_t i = 0, n = Rf_xlength(dataframe); i < n; i++)
  {
    RObject column = VECTOR_ELT(dataframe, i);

    // ensure name is converted to appropriate encoding
    PyObjectPtr name(as_python_str(names[i]));

    int status = 0;

    if (OBJECT(column) != 0) {
      // An object with a class attribute, we dispatch to the S3 method
      // and continue to the next column.
      // see comment in r_to_py() for why indirection in constructor is needed.
      RObject ref_(r_convert_dataframe_column(column, convert));
      PyObjectRef ref(ref_);
      status = PyDict_SetItem(dict, name, ref.get());
      if (status != 0)
        throw PythonException(py_fetch_error());

      continue;
    }

    if (!is_convertible_to_numpy(column)) {
      // Not an atomic type supported by numpy, thus we use the default
      // cast engine and continue to the next column.
      PyObjectPtr value(r_to_py_cpp(column, convert));
      status = PyDict_SetItem(dict, name, value);

      if (status != 0)
        throw PythonException(py_fetch_error());

      continue;
    }

    // We are sure it's an atomic vector:
    // Atomic values STRSXP, INTSXP, REALSXP and CPLSXP
    if (!nullable_dtypes || TYPEOF(column) == CPLXSXP) {
      PyObjectPtr value(r_to_py_numpy(column, convert));
      status = PyDict_SetItem(dict, name, value);
    } else {
      // use Pandas nullable data types.
      PyObjectPtr value(r_to_py_pandas_nullable_series(column, convert));
      status = PyDict_SetItem(dict, name, value);
    }

    if (status != 0)
      throw PythonException(py_fetch_error());
  }

  return py_ref(dict.detach(), convert);
}

namespace {

PyObject* r_convert_date_impl(PyObject* datetime,
                              Date date)
{

  PyObjectPtr py_date(PyObject_CallMethod(
      datetime, "date", "iii",
      static_cast<int>(date.getYear()),
      static_cast<int>(date.getMonth()),
      static_cast<int>(date.getDay())));

  if (py_date == NULL)
    throw PythonException(py_fetch_error());

  return py_date.detach();
}

} // end anonymous namespace

// [[Rcpp::export]]
PyObjectRef r_convert_date(DateVector dates, bool convert) {

  GILScope _gil;
  PyObjectPtr datetime(PyImport_ImportModule("datetime"));

  // short path for n == 1
  R_xlen_t n = dates.size();
  if (n == 1) {
    Date date = dates[0];
    return py_ref(r_convert_date_impl(datetime, date), convert);
  }

  // regular path for n > 1
  PyObjectPtr list(PyList_New(n));

  for (R_xlen_t i = 0; i < n; ++i) {
    Date date = dates[i];
    PyList_SetItem(list, i, r_convert_date_impl(datetime, date));
  }

  return py_ref(list.detach(), convert);

}


// [[Rcpp::export]]
SEXP py_list_length(PyObjectRef x) {
  GILScope _gil;

  Py_ssize_t value;
  if (PyList_CheckExact(x))
    value = PyList_Size(x);
  else
    value = PyObject_Size(x);

  if (value <= static_cast<Py_ssize_t>(INT_MAX))
    return Rf_ScalarInteger((int) value);
  else
    return Rf_ScalarReal((double) value);
}

// [[Rcpp::export]]
SEXP py_len_impl(PyObjectRef x, SEXP defaultValue = R_NilValue) {
  GILScope _gil;
  PyObject *er_type, *er_value, *er_traceback;
  if (defaultValue != R_NilValue)
    PyErr_Fetch(&er_type, &er_value, &er_traceback);

  Py_ssize_t value = PyObject_Size(x);
  if (value == -1) {
   // object is missing a `__len__` method, or a `__len__` method that
   // intentionally raises an Exception
    if (defaultValue == R_NilValue) {
      throw PythonException(py_fetch_error());
    } else {
      PyErr_Restore(er_type, er_value, er_traceback);
      return defaultValue;
    }
  }

  if (value <= static_cast<Py_ssize_t>(INT_MAX))
    return Rf_ScalarInteger((int) value);
  else
    return Rf_ScalarReal((double) value);
}

// [[Rcpp::export]]
SEXP py_bool_impl(PyObjectRef x, bool silent = false) {
  GILScope _gil;
  int result;
  if(silent) {
    PyErrorScopeGuard _g;

    // evaluate Python `not not x`
    result = PyObject_IsTrue(x);
    // result==-1 should only happen if the object has a
    // __bool__() method that intentionally raises an exception.
    if(result == -1)
      result = NA_LOGICAL;

  } else {

    result = PyObject_IsTrue(x);
    if(result == -1)
      throw PythonException(py_fetch_error());

  }

  return Rf_ScalarLogical(result);
}


// [[Rcpp::export]]
SEXP py_has_method(PyObjectRef object, const std::string& name) {
  GILScope _gil;
  PyObject* object_ = object.get(); // ensure python initialized, module proxy resolved

  PyObjectPtr attr(PyObject_GetAttrString(object_, name.c_str()));
  if (attr.is_null()) {
    PyErr_Clear();
    return Rf_ScalarLogical(false);
  }

  int result = PyMethod_Check(attr);
  return Rf_ScalarLogical(result);
}


//' Unique identifer for Python object
//'
//' Get a globally unique identifier for a Python object.
//'
//' @note In the current implementation of CPython this is the
//'  memory address of the object.
//'
//' @param object Python object
//'
//' @return Unique identifer (as string) or `NULL`
//'
//' @export
// [[Rcpp::export]]
SEXP py_id(PyObjectRef object) {
  if (py_is_null_xptr(object))
    return R_NilValue;
  GILScope _gil;

  std::stringstream id;
  id << (uintptr_t) object.get();

  return CharacterVector({id.str()});
}


// [[Rcpp::export]]
PyObjectRef py_capsule(SEXP x) {
  GILScope _gil;

  return py_ref(py_capsule_new(x), false);
}


// [[Rcpp::export]]
PyObjectRef py_slice(SEXP start = R_NilValue, SEXP stop = R_NilValue, SEXP step = R_NilValue) {
  GILScope _gil;

  PyObjectPtr start_, stop_, step_;

  if (start != R_NilValue)
    start_.assign(PyLong_FromLong(Rf_asInteger(start)));
  if (stop != R_NilValue)
    stop_.assign(PyLong_FromLong(Rf_asInteger(stop)));
  if (step != R_NilValue)
    step_.assign(PyLong_FromLong(Rf_asInteger(step)));

  PyObject* out(PySlice_New(start_, stop_, step_));
  if (out == NULL)
    throw PythonException(py_fetch_error());
  return py_ref(out, false);
}


//' @rdname iterate
//' @export
// [[Rcpp::export]]
SEXP as_iterator(SEXP x) {
  GILScope _gil;

  // If already inherits from iterator, return as is
  if (inherits2(x, "python.builtin.iterator"))
    return x;

  PyObject* iterable;
  PyObjectPtr iterable_ptr;
  bool convert;

  if (is_py_object(x)) {
    // unwrap PyObjectRef / Python objects
    PyObjectRef ref(x, false);
    iterable = ref.get();
    convert = ref.convert();
  }
  else {
    // If not already a py object, cast with r_to_py()
    iterable = r_to_py(x, true);   // returns a new ref
    iterable_ptr.assign(iterable); // decref on scope exit
    convert = true;
  }

  PyObject* iterator(PyObject_GetIter(iterable)); // returns new ref
  if(iterator == NULL)
    throw PythonException(py_fetch_error());

  return py_ref(iterator, convert);
}


// [[Rcpp::export]]
SEXP py_iter_next(PyObjectRef iterator, RObject completed) {
  GILScope _gil;

  if(!PyIter_Check(iterator))
    stop("object is not an iterator");

  PyObjectPtr item(PyIter_Next(iterator));
  if (item.is_null()) {

    // null could mean that iteraton is done so we check to
    // ensure that an error actually occrred
    if (PyErr_Occurred())
      throw PythonException(py_fetch_error());

    // if there wasn't an error then return the 'completed' sentinel
    return completed;

  } else {

    // return R object (PyObjectRef or converted obj)
    return py_to_r(item, iterator.convert());
  }
}


// Traverse a Python iterator or generator

// [[Rcpp::export]]
SEXP py_iterate(PyObjectRef x, Function f, bool simplify = true) {

  GILScope _gil;

  SEXP out;
  { // open scope so we can invoke c++ destructors before
    // calling UNPROTECT on the out object

  // List to return
  std::vector<RObject> list;

  // get the iterator
  PyObjectPtr iterator(PyObject_GetIter(x));
  if (iterator.is_null())
    throw PythonException(py_fetch_error());

  bool convert(x.convert());
  // loop over it
  while (true) {

    // check next item
    PyObjectPtr item(PyIter_Next(iterator));
    if (item.is_null()) {
      // null return means either iteration is done or
      // that there is an error
      if (PyErr_Occurred())
        throw PythonException(py_fetch_error());
      else
        break;
    }

    // get sexp (PyObjectRef or converted r obj)
    SEXP ret = py_to_r(item, convert);
    list.push_back(f(ret));
  }

  auto list_size = list.size();

  if (list_size == 0) {
    out = PROTECT(Rf_allocVector(VECSXP, 0));
    goto done;
  }

  int outType;
  if (simplify && convert)
  {
      // iterate over `list` to see if we have a common SEXP atomic type and length
      outType = TYPEOF(list[0]);
      switch (outType)
      {
      case INTSXP:
      case REALSXP:
      case LGLSXP:
      case STRSXP:
      case CPLXSXP:
          // iterate over list, see if all items are scalar atomics of the same type
          // If not, break early and return a list
          for (size_t i = 1; i < list_size; i++)
          {
              SEXP item = list[i];
              if (TYPEOF(item) != outType ||
                  OBJECT(item) ||
                  Rf_length(item) != 1)
              {
                  outType = VECSXP;
                  break;
              }
          }
          break;
      default:
          outType = VECSXP;
      }
  }
  else
  {
      outType = VECSXP;
  }
  // allocate an R object of type outType
  // copy over the list elements
  out = PROTECT(Rf_allocVector(outType, list_size));
  switch (outType)
  {
  case LGLSXP: {
      int *pout = LOGICAL(out);
      for (size_t i = 0; i < list_size; i++)
          pout[i] = LOGICAL_ELT(list[i], 0);
      break;
  }

  case INTSXP: {
      int *pout = INTEGER(out);
      for (size_t i = 0; i < list_size; i++)
          pout[i] = INTEGER_ELT(list[i], 0);
      break;
  }

  case REALSXP: {
      double *pout = REAL(out);
      for (size_t i = 0; i < list_size; i++)
          pout[i] = REAL_ELT(list[i], 0);
      break;
  }

  case CPLXSXP: {
      Rcomplex *pout = COMPLEX(out);
      for (size_t i = 0; i < list_size; i++)
          pout[i] = COMPLEX_ELT(list[i], 0);
      break;
  }

  case STRSXP: {
      for (size_t i = 0; i < list_size; i++)
          SET_STRING_ELT(out, i, STRING_ELT(list[i], 0));
      break;
  }

  case VECSXP: {
      for (size_t i = 0; i < list_size; i++)
          SET_VECTOR_ELT(out, i, list[i]);
      break;
  }

  default:
      // should never happen
      Rf_error("Internal error: unexpected type encountered in py_iterate");
  }
  }
  done:
  UNPROTECT(1);
  return out;
}


bool try_py_resolve_module_proxy(SEXP proxy) {
  Rcpp::Environment pkgEnv = Rcpp::Environment::namespace_env("reticulate");
  Rcpp::Function py_resolve_module_proxy = pkgEnv["py_resolve_module_proxy"];
  return py_resolve_module_proxy(proxy);
}



SEXP py_exception_as_condition(PyObject* object, SEXP refenv) {
  static SEXP names = []() {
    SEXP names = Rf_allocVector(STRSXP, 2);
    R_PreserveObject(names);
    SET_STRING_ELT(names, 0, Rf_mkChar("message"));
    SET_STRING_ELT(names, 1, Rf_mkChar("call"));
    return names;
  }();
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));

  SET_VECTOR_ELT(out, 0, Rcpp::wrap(conditionMessage_from_py_exception(object)));
  PyObject* call = py_get_attr(object, "call");
  if(call != NULL)
    SET_VECTOR_ELT(out, 1, py_to_r(call, true));

  Rf_setAttrib(out, R_NamesSymbol, names);
  Rf_setAttrib(out, R_ClassSymbol, Rf_getAttrib(refenv, R_ClassSymbol));
  Rf_setAttrib(out, sym_py_object, refenv);
  UNPROTECT(1);
  return out;
}


// [[Rcpp::export]]
bool py_allow_threads_impl(bool allow = true) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  if (allow) {
    PyGILState_Release(PyGILState_UNLOCKED);
  } else {
    PyGILState_Release(PyGILState_LOCKED);
  }
  return gstate == PyGILState_UNLOCKED;
}
