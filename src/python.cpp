
#include "libpython.h"

#define RCPP_NO_MODULES
#define RCPP_NO_SUGAR

#include <Rcpp.h>
using namespace Rcpp;

#include "signals.h"
#include "reticulate_types.h"

#include "event_loop.h"
#include "tinythread.h"

#include <fstream>
#include <time.h>

#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

using namespace reticulate::libpython;

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

// a simplified version of loadSymbol adopted from libpython.cpp
void loadSymbol(void* pLib, const std::string& name, void** ppSymbol)
{
  *ppSymbol = NULL;
#ifdef _WIN32
  *ppSymbol = (void*) ::GetProcAddress((HINSTANCE)pLib, name.c_str());
#else
  *ppSymbol = ::dlsym(pLib, name.c_str());
#endif
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
  if (!haveNumPy())
    return false;
  else
    return PyArray_Check(object);
}

bool isPyArrayScalar(PyObject* object) {
  if (!haveNumPy())
    return false;
  else
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
std::string py_fetch_error();

// wrap an R object in a longer-lived python object "capsule"
SEXP py_capsule_read(PyObject* capsule) {

  SEXP object = (SEXP) PyCapsule_GetPointer(capsule, NULL);
  if (object == NULL)
    stop(py_fetch_error());

  return object;

}

void py_capsule_free(PyObject* capsule) {

  SEXP object = py_capsule_read(capsule);
  if (object != R_NilValue)
    R_ReleaseObject(object);

}

PyObject* py_capsule_new(SEXP object) {

  if (object != R_NilValue)
    ::R_PreserveObject(object);

  return PyCapsule_New(
    (void*) object,
    NULL,
    py_capsule_free);

}

PyObject* py_get_attr(PyObject* object, const std::string& name) {

  if (PyObject_HasAttrString(object, name.c_str()))
    return PyObject_GetAttrString(object, name.c_str());
  else
    return NULL;

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
    if (object_ != NULL)
      Py_DecRef((PyObject*) object_);
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

PyObject* PyUnicode_AsBytes(PyObject* str) {
  return PyUnicode_AsEncodedString(str, "utf-8", "ignore");
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
    stop(py_fetch_error());

  return std::string(buffer, length);
}

#define as_utf8_r_string(str) Rcpp::String(as_std_string(str))

PyObject* as_python_str(SEXP strSEXP) {
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
// guaranteed to return NPY_BOOL, NPY_LONG, NPY_DOUBLE, or NPY_CDOUBLE
// (throws an exception if it's unable to return one of these types)
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
    break;

    // unsupported
  default:
    stop("Conversion from numpy array type %d is not supported", typenum);
    break;
  }

  return typenum;
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

std::string as_r_class(PyObject* classPtr) {

  PyObjectPtr namePtr(PyObject_GetAttrString(classPtr, "__name__"));
  std::ostringstream ostr;
  std::string module;

  if (PyObject_HasAttrString(classPtr, "__module__")) {
    PyObjectPtr modulePtr(PyObject_GetAttrString(classPtr, "__module__"));
    module = as_std_string(modulePtr) + ".";
    std::string builtin("__builtin__");
    if (module.find(builtin) == 0)
      module.replace(0, builtin.length(), "python.builtin");
    std::string builtins("builtins");
    if (module.find(builtins) == 0)
      module.replace(0, builtins.length(), "python.builtin");
  } else {
    module = "python.builtin.";
  }

  ostr << module << as_std_string(namePtr);
  return ostr.str();

}

std::vector<std::string> py_class_names(PyObject* object) {

  // class
  PyObjectPtr classPtr(PyObject_GetAttrString(object, "__class__"));
  if (classPtr.is_null())
    stop(py_fetch_error());

  // call inspect.getmro to get the class and it's bases in
  // method resolution order
  PyObjectPtr inspect(py_import("inspect"));
  if (inspect.is_null())
    stop(py_fetch_error());

  PyObjectPtr getmro(PyObject_GetAttrString(inspect, "getmro"));
  if (getmro.is_null())
    stop(py_fetch_error());

  PyObjectPtr classes(PyObject_CallFunctionObjArgs(getmro, classPtr.get(), NULL));
  if (classes.is_null())
    stop(py_fetch_error());

  // start adding class names
  std::vector<std::string> classNames;

  // add the bases to the R class attribute
  Py_ssize_t len = PyTuple_Size(classes);
  for (Py_ssize_t i = 0; i < len; i++) {
    PyObject* base = PyTuple_GetItem(classes, i); // borrowed
    classNames.push_back(as_r_class(base));
  }

  // return constructed class names
  return classNames;

}

// wrap a PyObject
PyObjectRef py_ref(PyObject* object,
                   bool convert,
                   const std::string& extraClass = "")
{
  // wrap
  PyObjectRef ref(object, convert);

  // class attribute
  std::vector<std::string> attrClass;

  // add extra class if requested
  if (!extraClass.empty() && std::find(attrClass.begin(),
                        attrClass.end(),
                        extraClass) == attrClass.end()) {
    attrClass.push_back(extraClass);
  }

  // register R classes
  if (PyObject_HasAttrString(object, "__class__")) {
    std::vector<std::string> classNames = py_class_names(object);
    attrClass.insert(attrClass.end(), classNames.begin(), classNames.end());
  }

  // add python.builtin.object if we don't already have it
  if (std::find(attrClass.begin(), attrClass.end(), "python.builtin.object") == attrClass.end()) {
    attrClass.push_back("python.builtin.object");
  }

  // apply class filter
  Rcpp::Environment pkgEnv = Rcpp::Environment::namespace_env("reticulate");
  Rcpp::Function py_filter_classes = pkgEnv["py_filter_classes"];
  attrClass = as< std::vector<std::string> >(py_filter_classes(attrClass));

  // set classes
  ref.attr("class") = attrClass;

  // return ref
  return ref;

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


class LastError {

public:

  bool empty() const { return type_.empty(); }

  std::string type() const { return type_; }
  void setType(const std::string& type) { type_ = type; }

  std::string value() const { return value_; }
  void setValue(const std::string& value) { value_ = value; }

  std::string message() const { return message_; }
  void setMessage(const std::string& message) { message_ = message; }

  std::vector<std::string> traceback() const { return traceback_; };
  void setTraceback(const std::vector<std::string>& traceback) {
    traceback_ = traceback;
  }

  void clear() {
    type_.clear();
    value_.clear();
    traceback_.clear();
    message_.clear();
  }

private:
  std::string type_;
  std::string value_;
  std::vector<std::string> traceback_;
  std::string message_;
};

LastError s_lastError;

//' Get or clear the last Python error encountered
//'
//' @return For `py_last_error()`, a list with the type, value,
//' and traceback for the last Python error encountered (can be
//' `NULL` if no error has yet been encountered).
//'
//' @export
// [[Rcpp::export]]
SEXP py_last_error() {
  if (s_lastError.empty()) {
    return R_NilValue;
  } else {
    List lastError;
    lastError["type"] = s_lastError.type();
    lastError["value"] = s_lastError.value();
    lastError["traceback"] = s_lastError.traceback();
    lastError["message"] = s_lastError.message();
    return lastError;
  }
}

//' @rdname py_last_error
//' @export
// [[Rcpp::export]]
void py_clear_last_error() {
  s_lastError.clear();
}


std::string py_fetch_error_type(const PyObjectPtr& pExcType) {

  if (pExcType.is_null())
    return std::string();

  PyObjectPtr pStr(PyObject_GetAttrString(pExcType, "__name__"));
  return as_std_string(pStr);

}

std::string py_fetch_error_value(const PyObjectPtr& pExcValue) {

  if (pExcValue.is_null())
    return std::string();

  PyObjectPtr pStr(PyObject_Str(pExcValue));
  return as_std_string(pStr);

}

void py_fetch_error_traceback(PyObject* excTraceback,
                              std::vector<std::string>* pTraceback)
{
  if (excTraceback == NULL)
    return;

  // invoke 'traceback.format_tb(<traceback>)'
  PyObjectPtr module(py_import("traceback"));
  if (module.is_null())
    return;

  PyObjectPtr format_tb(PyObject_GetAttrString(module, "format_tb"));
  if (format_tb.is_null())
    return;

  PyObjectPtr tb(PyObject_CallFunctionObjArgs(format_tb, excTraceback, NULL));
  if (tb.is_null())
    return;

  // get the traceback
  for (Py_ssize_t i = 0, n = PyList_Size(tb); i < n; i++)
    pTraceback->push_back(as_std_string(PyList_GetItem(tb, i)));

}

// get a string representing the last python error
std::string py_fetch_error() {

  // clear last error
  s_lastError.clear();

  // check whether this error was signaled via an interrupt.
  // the intention here is to catch cases where reticulate is running
  // Python code, an interrupt is signaled and caught by that code,
  // and then the associated error is returned. in such a case, we
  // want to forward that interrupt back to R so that the user is then
  // returned back to the top level.
  if (reticulate::signals::getPythonInterruptsPending()) {
    PyErr_Clear();
    reticulate::signals::setInterruptsPending(false);
    reticulate::signals::setPythonInterruptsPending(false);
    throw Rcpp::internal::InterruptedException();
  }

  // read and normalize error, exception
  PyObject *excType, *excValue, *excTraceback;
  PyErr_Fetch(&excType, &excValue, &excTraceback);
  PyErr_NormalizeException(&excType, &excValue, &excTraceback);

  // create object pointers (so they're freed on scope exit)
  PyObjectPtr pExcType(excType);
  PyObjectPtr pExcValue(excValue);
  PyObjectPtr pExcTraceback(excTraceback);

  // if we don't have any error information, return early
  if (pExcType.is_null() && pExcValue.is_null())
    return "<unknown error>";

  // build error text
  std::ostringstream oss;

  // get exception type
  std::string type = py_fetch_error_type(pExcType);

  // set type if we had it
  if (!type.empty()) {
    s_lastError.setType(type);
    oss << type << ": ";
  }

  // get exception value
  std::string value = py_fetch_error_value(pExcValue);
  if (!value.empty()) {
    s_lastError.setValue(value);
    oss << value;
  }

  // retrieve Python traceback
  std::vector<std::string> traceback;
  py_fetch_error_traceback(excTraceback, &traceback);
  s_lastError.setTraceback(traceback);

  if (traceback_enabled()) {
    std::size_t n = traceback.size();
    if (n > 0) {
      oss << "\n\nDetailed traceback:\n";
      for (std::size_t i = 0; i < n; i++)
        oss << traceback[i];
    }
  }

  // get final error string
  std::string error = oss.str();
  s_lastError.setMessage(error);

  // return error
  return error;
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

void set_string_element(SEXP rArray, int i, PyObject* pyStr) {
  std::string str = as_std_string(pyStr);
  cetype_t ce = PyUnicode_Check(pyStr) ? CE_UTF8 : CE_NATIVE;
  SEXP strSEXP = Rf_mkCharCE(str.c_str(), ce);
  SET_STRING_ELT(rArray, i, strSEXP);
}

bool py_equal(PyObject* x, const std::string& str) {

  PyObjectPtr pyStr(as_python_str(str));
  if (pyStr.is_null())
    stop(py_fetch_error());

  return PyObject_RichCompareBool(x, pyStr, Py_EQ) == 1;

}

bool is_pandas_na(PyObject* x) {

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


bool py_is_callable(PyObject* x) {
  return PyCallable_Check(x) == 1 || PyObject_HasAttrString(x, "__call__");
}

// [[Rcpp::export]]
PyObjectRef py_none_impl() {
  Py_IncRef(Py_None);
  return py_ref(Py_None, false);
}

// [[Rcpp::export]]
bool py_is_callable(PyObjectRef x) {
  if (x.is_null_xptr())
    return false;
  else
    return py_is_callable(x.get());
}


// convert a python object to an R object
SEXP py_to_r(PyObject* x, bool convert) {

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
  else if (PyList_Check(x)) {

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
        list[i] = py_to_r(PyList_GetItem(x, i), convert);
      return list;
    }
  }

  // tuple (but don't convert namedtuple as it's often a custom class)
  else if (PyTuple_Check(x) && !PyObject_HasAttrString(x, "_fields")) {
    Py_ssize_t len = PyTuple_Size(x);
    Rcpp::List list(len);
    for (Py_ssize_t i = 0; i<len; i++)
      list[i] = py_to_r(PyTuple_GetItem(x, i), convert);
    return list;
  }

  // dict
  else if (PyDict_Check(x)) {

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
  else if (isPyArray(x)) {

    // R array to return
    RObject rArray = R_NilValue;

    // get the array
    PyArrayObject* array = (PyArrayObject*) x;

    // get the dimensions -- treat 0-dim array (numpy scalar) as
    // a 1-dim for converstion to R (will end up with a single
    // element R vector)
    npy_intp len = PyArray_SIZE(array);
    int nd = PyArray_NDIM(array);
    IntegerVector dimsVector(nd);
    if (nd > 0) {
      npy_intp *dims = PyArray_DIMS(array);
      for (int i = 0; i<nd; i++)
        dimsVector[i] = dims[i];
    } else {
      dimsVector.push_back(1);
    }

    // determine the target type of the array
    int typenum = narrow_array_typenum(array);

    // cast it to a fortran array (PyArray_CastToType steals the descr)
    // (note that we will decref the copied array below)
    PyArray_Descr* descr = PyArray_DescrFromType(typenum);
    array = (PyArrayObject*) PyArray_CastToType(array, descr, NPY_ARRAY_FARRAY);
    if (array == NULL)
      stop(py_fetch_error());

    // ensure we release it within this scope
    PyObjectPtr ptrArray((PyObject*)array);

    // copy the data as required per-type
    switch(typenum) {

      case NPY_BOOL: {
        npy_bool* pData = (npy_bool*)PyArray_DATA(array);
        rArray = Rf_allocArray(LGLSXP, dimsVector);
        for (int i=0; i<len; i++)
          LOGICAL(rArray)[i] = pData[i];
        break;
      }

      case NPY_LONG: {
        npy_long* pData = (npy_long*)PyArray_DATA(array);
        rArray = Rf_allocArray(INTSXP, dimsVector);
        for (int i=0; i<len; i++)
          INTEGER(rArray)[i] = pData[i];
        break;
      }

      case NPY_DOUBLE: {
        npy_double* pData = (npy_double*)PyArray_DATA(array);
        rArray = Rf_allocArray(REALSXP, dimsVector);
        for (int i=0; i<len; i++)
          REAL(rArray)[i] = pData[i];
        break;
      }

      case NPY_CDOUBLE: {
        npy_complex128* pData = (npy_complex128*)PyArray_DATA(array);
        rArray = Rf_allocArray(CPLXSXP, dimsVector);
        for (int i=0; i<len; i++) {
          npy_complex128 data = pData[i];
          Rcomplex cpx;
          cpx.r = data.real;
          cpx.i = data.imag;
          COMPLEX(rArray)[i] = cpx;
        }
        break;
      }

      case NPY_STRING:
      case NPY_UNICODE: {
        PyObjectPtr itemFunc(PyObject_GetAttrString(ptrArray, "item"));
        if (itemFunc.is_null())
          stop(py_fetch_error());
        rArray = Rf_allocArray(STRSXP, dimsVector);
        RObject protectArray(rArray);
        for (int i=0; i<len; i++) {
          PyObjectPtr pyArgs(PyTuple_New(1));
          // PyTuple_SetItem steals reference to object created by PyInt_FromLong
          PyTuple_SetItem(pyArgs, 0, PyInt_FromLong(i));
          PyObjectPtr pyStr(PyObject_Call(itemFunc, pyArgs, NULL));
          if (pyStr.is_null()) {
            stop(py_fetch_error());
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
          if (!is_python_str(pData[i])) {
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
    }

    // return the R Array
    return rArray;

  }

  // check for numpy scalar
  else if (isPyArrayScalar(x)) {

    // determine the type to convert to
    PyArray_DescrPtr descrPtr(PyArray_DescrFromScalar(x));
    int typenum = narrow_array_typenum(descrPtr);
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
      return NumericVector::create(value);
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

  // callable
  else if (py_is_callable(x)) {

    // reference to underlying python object
    Py_IncRef(x);
    PyObjectRef pyFunc = py_ref(x, convert);

    // create an R function wrapper
    Rcpp::Environment pkgEnv = Rcpp::Environment::namespace_env("reticulate");
    Rcpp::Function py_callable_as_function = pkgEnv["py_callable_as_function"];
    Rcpp::Function f = py_callable_as_function(pyFunc, convert);

    // forward classes
    f.attr("class") = pyFunc.attr("class");

    // save reference to underlying py_object
    f.attr("py_object") = pyFunc;

    // return the R function
    return f;
  }

  // iterator/generator
  else if (PyObject_HasAttrString(x, "__iter__") &&
           (PyObject_HasAttrString(x, "next") ||
            PyObject_HasAttrString(x, "__next__"))) {

    // return it raw but add a class so we can create S3 methods for it
    Py_IncRef(x);
    return py_ref(x, true, std::string("python.builtin.iterator"));
  }

  // bytearray
  else if (PyByteArray_Check(x)) {

    if (PyByteArray_Size(x) == 0)
      return RawVector();

    return RawVector(
      PyByteArray_AsString(x),
      PyByteArray_AsString(x) + PyByteArray_Size(x));

  }

  // pandas array
  else if (is_pandas_na(x)) {
    return NumericVector::create(R_NaReal);
  }

  // default is to return opaque wrapper to python object. we pass convert = true
  // because if we hit this code then conversion has been either implicitly
  // or explicitly requested.
  else {
    Py_IncRef(x);
    return py_ref(x, true);
  }

}

// [[Rcpp::export]]
SEXP py_get_formals(PyObjectRef func) {

  PyObjectPtr inspect(py_import("inspect"));
  if (inspect.is_null())
    stop(py_fetch_error());

  PyObjectPtr get_signature(PyObject_GetAttrString(inspect.get(), "signature"));
  if (get_signature.is_null())
    stop(py_fetch_error());

  PyObjectPtr signature(PyObject_CallFunctionObjArgs(get_signature.get(), func.get(), NULL));
  if (signature.is_null())
    stop(py_fetch_error());

  PyObjectPtr param_dict(PyObject_GetAttrString(signature.get(), "parameters"));
  if (param_dict.is_null())
    stop(py_fetch_error());

  PyObjectPtr param_values(PyObject_GetAttrString(param_dict.get(), "values"));
  if (param_values.is_null())
    stop(py_fetch_error());

  PyObjectPtr params(PyObject_CallFunctionObjArgs(param_values.get(), NULL, NULL));
  if (params.is_null())
    stop(py_fetch_error());

  PyObjectPtr param_iter(PyObject_GetIter(params.get()));
  if (param_iter.is_null())
    stop(py_fetch_error());

  // Static properties of the Parameter class
  PyObjectPtr param_class(PyObject_GetAttrString(inspect.get(), "Parameter"));
  if (param_class.is_null())
    stop(py_fetch_error());

  PyObjectPtr empty_param(PyObject_GetAttrString(param_class.get(), "empty"));
  if (empty_param.is_null())
    stop(py_fetch_error());

  PyObjectPtr var_pos(PyObject_GetAttrString(param_class.get(), "VAR_POSITIONAL"));
  if (var_pos.is_null())
    stop(py_fetch_error());

  PyObjectPtr var_kw(PyObject_GetAttrString(param_class.get(), "VAR_KEYWORD"));
  if (var_kw.is_null())
    stop(py_fetch_error());

  PyObjectPtr kw_only(PyObject_GetAttrString(param_class.get(), "KEYWORD_ONLY"));
  if (kw_only.is_null())
    stop(py_fetch_error());

  Rcpp::Pairlist formals;
  bool var_encountered = false;
  while (true) {

    PyObjectPtr param(PyIter_Next(param_iter.get()));
    if (param.is_null())
      break;

    PyObjectPtr param_name(PyObject_GetAttrString(param.get(), "name"));
    if (param_name.is_null())
      stop(py_fetch_error());

    PyObjectPtr param_kind(PyObject_GetAttrString(param.get(), "kind"));
    if (param_kind.is_null())
      stop(py_fetch_error());

    PyObjectPtr param_default(PyObject_GetAttrString(param.get(), "default"));
    if (param_default.is_null())
      stop(py_fetch_error());

    // If we encounter our first kw_only param
    // without having encountered `*args` or `**kw`,
    // we insert `...` before the actual parameter.
    if (param_kind == kw_only && !var_encountered) {
      formals << Named("...", R_MissingArg);
      var_encountered = true;
    }

    // If we encounter the first of `*args` or `**kw`,
    // we insert `...` instead of a parameter.
    // foo(*args, b=1, **kw) -> foo(..., b=1)
    if (param_kind == var_pos || param_kind == var_kw) {
      if (!var_encountered) {
        formals << Named("...", R_MissingArg);
        var_encountered = true;
      }
    // For a parameter w/o default value, we insert `R_MissingArg`.
    // There is inspect.Parameter(..., *, default = Parameter.empty, ...)
    // so we check if param_kind != kw_only for this corner case.
    } else if (param_kind != kw_only && param_default == empty_param) {
      formals << Named(as_utf8_r_string(param_name.get()), R_MissingArg);
    // If we arrive here we have a parameter with default value.
    } else {
      // Here we could convert a subset of python objects to R defaults.
      // Plain values (numeric, character, NULL, ...) are stored as is,
      // variables, calls, ... are stored as `symbol` or `language`.
      formals << Named(as_utf8_r_string(param_name.get()), R_NilValue);
    }
  }

  return formals;

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
  IntegerVector dimensions = x.hasAttribute("dim")
    ? x.attr("dim")
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
  } else {
    stop("Matrix type cannot be converted to python (only integer, "
           "numeric, complex, logical, and character matrixes can be "
           "converted");
  }

  int flags = NPY_ARRAY_FARRAY_RO;

  // because R logical vectors are just ints under the
  // hood, we need to explicitly construct a boolean
  // vector for our Python array. note that the created
  // array will own the data so we do not free it after
  if (typenum == NPY_BOOL) {
    R_xlen_t n = XLENGTH(sexp);
    bool* converted = (bool*) PyArray_malloc(n * sizeof(bool));
    for (R_xlen_t i = 0; i < n; i++)
      converted[i] = LOGICAL(sexp)[i];
    data = converted;
    flags |= NPY_ARRAY_OWNDATA;
  }

  // create the matrix
  PyObject* array = PyArray_New(&PyArray_Type,
                                nd,
                                &(dims[0]),
                                typenum,
                                NULL,
                                data,
                                0,
                                flags,
                                NULL);

  // check for error
  if (array == NULL)
    stop(py_fetch_error());

  // if this is a character vector we need to convert and set the elements,
  // otherwise the memory is shared with the underlying R vector
  if (type == STRSXP) {
    void** pData = (void**)PyArray_DATA((PyArrayObject*)array);
    R_xlen_t len = Rf_xlength(x);
    for (R_xlen_t i = 0; i<len; i++) {
      PyObject* pyStr = as_python_str(STRING_ELT(x, i));
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
        stop(py_fetch_error());
    } else {
      PyArray_BASE(array) = capsule.detach();
    }
  }

  // return it
  return array;

}

PyObject* r_to_py_cpp(RObject x, bool convert);

PyObject* r_to_py(RObject x, bool convert) {

  // if the object bit is not set, we can skip R dispatch
  if (OBJECT(x) == 0)
    return r_to_py_cpp(x, convert);

  // get a reference to the R version of r_to_py
  Rcpp::Environment pkgEnv = Rcpp::Environment::namespace_env("reticulate");
  Rcpp::Function r_to_py_fn = pkgEnv["r_to_py"];

  // call the R version and hold the return value in a PyObjectRef (SEXP wrapper)
  // this object will be released when the function returns
  PyObjectRef ref(r_to_py_fn(x, convert));

  // get the underlying Python object and call Py_IncRef before returning it
  // this allows this function to provide the same memory semantics as the
  // previous C++ version of r_to_py (which is now r_to_py_cpp), which always
  // calls Py_IncRef on Python objects before returning them
  PyObject* obj = ref.get();
  Py_IncRef(obj);

  // return the Python object
  return obj;
}

// Python capsule wrapping an R's external pointer object
static void free_r_extptr_capsule(PyObject* capsule) {
  SEXP sexp = (SEXP)PyCapsule_GetContext(capsule);
  ::R_ReleaseObject(sexp);
}

static PyObject* r_extptr_capsule(SEXP sexp) {

  // underlying pointer
  void* ptr = R_ExternalPtrAddr(sexp);
  if (ptr == NULL)
    stop("Invalid pointer");

  ::R_PreserveObject(sexp);

  PyObject* capsule = PyCapsule_New(ptr, NULL, free_r_extptr_capsule);
  PyCapsule_SetContext(capsule, (void*)sexp);
  return capsule;

}

// convert an R object to a python object (the returned object
// will have an active reference count on it)
PyObject* r_to_py_cpp(RObject x, bool convert) {

  int type = x.sexp_type();
  SEXP sexp = x.get__();

  // NULL becomes python None
  // (Py_IncRef since PyTuple_SetItem will steal the passed reference)
  if (x.isNULL()) {
    Py_IncRef(Py_None);
    return Py_None;
  }

  // use py_object attribute if we have it
  if (x.hasAttribute("py_object")) {
    Rcpp::RObject py_object = x.attr("py_object");
    PyObjectRef obj = as<PyObjectRef>(py_object);
    Py_IncRef(obj.get());
    return obj.get();
  }

  // pass python objects straight through (Py_IncRef since returning this
  // creates a new reference from the caller)
  if (x.inherits("python.builtin.object")) {
    PyObjectRef obj = as<PyObjectRef>(sexp);
    Py_IncRef(obj.get());
    return obj.get();
  }

  // convert arrays and matrixes to numpy (throw error if numpy not available)
  if (x.hasAttribute("dim") && requireNumPy()) {
    return r_to_py_numpy(x, convert);
  }

  // integer (pass length 1 vectors as scalars, otherwise pass list)
  if (type == INTSXP) {

    // handle scalars
    if (LENGTH(sexp) == 1) {
      int value = INTEGER(sexp)[0];
      return PyInt_FromLong(value);
    }

    PyObjectPtr list(PyList_New(LENGTH(sexp)));
    for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
      int value = INTEGER(sexp)[i];
      // NOTE: reference to added value is "stolen" by the list
      int res = PyList_SetItem(list, i, PyInt_FromLong(value));
      if (res != 0)
        stop(py_fetch_error());
    }

    return list.detach();

  }

  // numeric (pass length 1 vectors as scalars, otherwise pass list)
  if (type == REALSXP) {

    // handle scalars
    if (LENGTH(sexp) == 1) {
      double value = REAL(sexp)[0];
      return PyFloat_FromDouble(value);
    }

    PyObjectPtr list(PyList_New(LENGTH(sexp)));
    for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
      double value = REAL(sexp)[i];
      // NOTE: reference to added value is "stolen" by the list
      int res = PyList_SetItem(list, i, PyFloat_FromDouble(value));
      if (res != 0)
        stop(py_fetch_error());
    }

    return list.detach();

  }

  // complex (pass length 1 vectors as scalars, otherwise pass list)
  if (type == CPLXSXP) {

    // handle scalars
    if (LENGTH(sexp) == 1) {
      Rcomplex cplx = COMPLEX(sexp)[0];
      return PyComplex_FromDoubles(cplx.r, cplx.i);
    }

    PyObjectPtr list(PyList_New(LENGTH(sexp)));
    for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
      Rcomplex cplx = COMPLEX(sexp)[i];
      // NOTE: reference to added value is "stolen" by the list
      int res = PyList_SetItem(list, i, PyComplex_FromDoubles(cplx.r, cplx.i));
      if (res != 0)
        stop(py_fetch_error());
    }

    return list.detach();

  }

  // logical (pass length 1 vectors as scalars, otherwise pass list)
  if (type == LGLSXP) {

    // handle scalars
    if (LENGTH(sexp) == 1) {
      int value = LOGICAL(sexp)[0];
      return PyBool_FromLong(value);
    }

    PyObjectPtr list(PyList_New(LENGTH(sexp)));
    for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
      int value = LOGICAL(sexp)[i];
      // NOTE: reference to added value is "stolen" by the list
      int res = PyList_SetItem(list, i, PyBool_FromLong(value));
      if (res != 0)
        stop(py_fetch_error());
    }

    return list.detach();

  }

  // character (pass length 1 vectors as scalars, otherwise pass list)
  if (type == STRSXP) {

    // handle scalars
    if (LENGTH(sexp) == 1) {
      return as_python_str(STRING_ELT(sexp, 0));
    }

    PyObjectPtr list(PyList_New(LENGTH(sexp)));
    for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
      // NOTE: reference to added value is "stolen" by the list
      int res = PyList_SetItem(list, i, as_python_str(STRING_ELT(sexp, i)));
      if (res != 0)
        stop(py_fetch_error());
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

    // create a dict for names
    if (x.hasAttribute("names")) {
      PyObjectPtr dict(PyDict_New());
      CharacterVector names = x.attr("names");
      SEXP namesSEXP = names;
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        const char* name = Rf_translateChar(STRING_ELT(namesSEXP, i));
        PyObjectPtr item(r_to_py(RObject(VECTOR_ELT(sexp, i)), convert));
        int res = PyDict_SetItemString(dict, name, item);
        if (res != 0)
          stop(py_fetch_error());
      }

      return dict.detach();

    }

    // create a list if there are no names
    PyObjectPtr list(PyList_New(LENGTH(sexp)));
    for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
      PyObject* item = r_to_py(RObject(VECTOR_ELT(sexp, i)), convert);
      // NOTE: reference to added value is "stolen" by the list
      int res = PyList_SetItem(list, i, item);
      if (res != 0)
        stop(py_fetch_error());
    }

    return list.detach();

  }

  if (type == CLOSXP) {

    // create an R object capsule for the R function
    PyObjectPtr capsule(py_capsule_new(x));
    PyCapsule_SetContext(capsule, (void*)convert);

    // check for a py_function_name attribute
    PyObjectPtr pyFunctionName(r_to_py(x.attr("py_function_name"), convert));

    // create the python wrapper function
    PyObjectPtr module(py_import("rpytools.call"));
    if (module.is_null())
      stop(py_fetch_error());

    PyObjectPtr func(PyObject_GetAttrString(module, "make_python_function"));
    if (func.is_null())
      stop(py_fetch_error());

    PyObjectPtr wrapper(
        PyObject_CallFunctionObjArgs(
          func,
          capsule.get(),
          pyFunctionName.get(),
          NULL));

    if (wrapper.is_null())
      stop(py_fetch_error());

    // return the wrapper
    return wrapper.detach();

  }

  // externalptr
  if (type == EXTPTRSXP) {
    return r_extptr_capsule(sexp);
  }

  // unhandled type
  Rcpp::print(sexp);
  stop("Unable to convert R object to Python type");

}

// [[Rcpp::export]]
PyObjectRef r_to_py_impl(RObject object, bool convert) {
  return py_ref(r_to_py_cpp(object, convert), convert);
}

// custom module used for calling R functions from python wrappers



extern "C" PyObject* call_r_function(PyObject *self, PyObject* args, PyObject* keywords)
{
  // the first argument is always the capsule containing the R function to call
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
    for (Py_ssize_t index = 0; index<len; index++) {
      PyObject* item = PyTuple_GetItem(funcArgs, index); // borrowed
      Py_IncRef(item);
      rArgs.push_back(py_ref(item, convert));
    }
  }

  // get keyword arguments
  List rKeywords;

  if (keywords != NULL) {

    if (convert) {
      rKeywords = py_to_r(keywords, convert);
    } else {

      PyObject *key, *value;
      Py_ssize_t pos = 0;

      // NOTE: PyDict_Next uses borrowed references,
      // so anything we return should be Py_IncRef'd
      while (PyDict_Next(keywords, &pos, &key, &value)) {
        PyObjectPtr str(PyObject_Str(key));
        Py_IncRef(value);
        rKeywords[as_std_string(str)] = py_ref(value, convert);
      }
    }

  }

  // combine positional and keyword arguments
  Function append("append");
  rArgs = append(rArgs, rKeywords);

  // Some special constants for various special error conditions
  // (NOTE: these are also defined in call.py so must be changed in both places)
  const char* const kErrorKey = "F4B07A71E0ED40469929658827023424";
  const char* const kInterruptError = "E04414EDEA17488B93FE2AE30F1F67AF";

  // call the R function
  std::string err;
  try {
    Function doCall("do.call");
    RObject result = doCall(rFunction, rArgs);
    return r_to_py(result, convert);
  } catch(const Rcpp::internal::InterruptedException& e) {
    err = kInterruptError;
  } catch(const std::exception& e) {
    err = e.what();
  } catch(...) {
    err = "(Unknown exception occurred)";
  }

  // ...we won't reach this code unless an error occurred

  // Return a special named list which the caller transforms into a python error
  PyObjectPtr errorDict(PyDict_New());
  PyObjectPtr errorMsg(as_python_str(err));
  PyDict_SetItemString(errorDict, kErrorKey, errorMsg);
  return errorDict.detach();
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


extern "C" PyObject* call_python_function_on_main_thread(
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

  // return none
  Py_IncRef(Py_None);
  return Py_None;
}


PyMethodDef RPYCallMethods[] = {
  { "call_r_function", (PyCFunction)call_r_function,
    METH_VARARGS | METH_KEYWORDS, "Call an R function" },
  { "call_python_function_on_main_thread", (PyCFunction)call_python_function_on_main_thread,
    METH_VARARGS | METH_KEYWORDS, "Call a Python function on the main thread" },
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

// forward declare py_run_file
PyObjectRef py_run_file_impl(const std::string& file);

// [[Rcpp::export]]
void py_activate_virtualenv(const std::string& script)
{
  // get main dict
  PyObject* main = PyImport_AddModule("__main__");
  PyObject* mainDict = PyModule_GetDict(main);

  // inject __file__
  PyObjectPtr file(as_python_str(script));
  int res = PyDict_SetItemString(mainDict, "__file__", file);
  if (res != 0)
    stop(py_fetch_error());

  // read the code in the script
  std::ifstream ifs(script.c_str());
  if (!ifs)
    stop("Unable to open file '%s' (does it exist?)", script);
  std::string code((std::istreambuf_iterator<char>(ifs)),
                   (std::istreambuf_iterator<char>()));

  // run string
  PyObjectPtr runRes(PyRun_StringFlags(code.c_str(), Py_file_input, mainDict, NULL, NULL));
  if (runRes.is_null())
    stop(py_fetch_error());
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

  List info;

  // read Python program path
  std::string python_path;
  if (Py_GetVersion()[0] >= '3') {
    loadSymbol(pLib, "Py_GetProgramFullPath", (void**) &Py_GetProgramFullPath);
    const std::wstring wide_python_path(Py_GetProgramFullPath());
    python_path = to_string(wide_python_path);
    info["python"] = python_path;
  } else {
    loadSymbol(pLib, "Py_GetProgramFullPath", (void**) &Py_GetProgramFullPath_v2);
    python_path = Py_GetProgramFullPath_v2();
    info["python"] = python_path;
  }

  // read libpython file path
  if (strcmp(python_path.c_str(), dinfo.dli_fname) == 0) {
    // if the library is the same as the executable, it's probably a PIE.
    // Any consequent dlopen on the PIE may fail, return NA to indicate this.
    info["libpython"] = Rf_ScalarString(R_NaString);
  } else {
    info["libpython"] = dinfo.dli_fname;
  }

  return info;

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
void py_initialize(const std::string& python,
                   const std::string& libpython,
                   const std::string& pythonhome,
                   const std::string& virtualenv_activate,
                   bool python3,
                   bool interactive,
                   const std::string& numpy_load_error) {

  // set python3 and interactive flags
  s_isPython3 = python3;
  s_isInteractive = interactive;

  // load the library
  std::string err;
  if (!libPython().load(libpython, is_python3(), &err))
    stop(err);

  if (is_python3()) {

    // set program name
    s_python_v3 = to_wstring(python);
    Py_SetProgramName_v3(const_cast<wchar_t*>(s_python_v3.c_str()));

    // set program home
    s_pythonhome_v3 = to_wstring(pythonhome);
    Py_SetPythonHome_v3(const_cast<wchar_t*>(s_pythonhome_v3.c_str()));

    if (Py_IsInitialized()) {
      // if R is embedded in a python environment, rpycall has to be loaded as a regular
      // module.
      PyImport_AddModule("rpycall");
      PyDict_SetItemString(PyImport_GetModuleDict(), "rpycall", initializeRPYCall());

    } else {
      // add rpycall module
      PyImport_AppendInittab("rpycall", &initializeRPYCall);

      // initialize python
      Py_Initialize();
    }

    const wchar_t *argv[1] = {s_python_v3.c_str()};
    PySys_SetArgv_v3(1, const_cast<wchar_t**>(argv));

  } else {

    // set program name
    s_python = python;
    Py_SetProgramName(const_cast<char*>(s_python.c_str()));

    // set program home
    s_pythonhome = pythonhome;
    Py_SetPythonHome(const_cast<char*>(s_pythonhome.c_str()));

    if (!Py_IsInitialized()) {
      // initialize python
      Py_Initialize();
    }

    // add rpycall module
    Py_InitModule4("rpycall", RPYCallMethods, (char *)NULL, (PyObject *)NULL,
                      _PYTHON_API_VERSION);

    const char *argv[1] = {s_python.c_str()};
    PySys_SetArgv(1, const_cast<char**>(argv));
  }

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
  std::string tracems_env = as<std::string>(sysGetEnv("RETICULATE_DUMP_STACK_TRACE", 0));
  int tracems = ::atoi(tracems_env.c_str());
  if (tracems > 0)
    trace_thread_init(tracems);

  // poll for events while executing python code
  reticulate::event_loop::initialize();

}

// [[Rcpp::export]]
void py_finalize() {

  // multiple calls to PyFinalize are likely to cause problems so
  // we comment this out to play better with other packages that include
  // python embedding code.

  // ::Py_Finalize();
}

// [[Rcpp::export]]
bool py_is_none(PyObjectRef x) {
  return py_is_none(x.get());
}

// [[Rcpp::export]]
bool py_compare_impl(PyObjectRef a, PyObjectRef b, const std::string& op) {

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
    stop(py_fetch_error());
  else
    return res == 1;
}

// [[Rcpp::export]]
CharacterVector py_str_impl(PyObjectRef x) {

  if (!is_python_str(x)) {

    PyObjectPtr str(PyObject_Str(x));
    if (str.is_null())
      stop(py_fetch_error());

    return CharacterVector::create(as_utf8_r_string(str));

  }

  return CharacterVector::create(as_utf8_r_string(x));

}

// [[Rcpp::export]]
void py_print(PyObjectRef x) {
  CharacterVector out = py_str_impl(x);
  Rf_PrintValue(out);
  Rcout << std::endl;
}

// [[Rcpp::export]]
bool py_is_function(PyObjectRef x) {
  return PyFunction_Check(x) == 1;
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
void py_validate_xptr(PyObjectRef x) {
  if (py_is_null_xptr(x)) {
    stop("Object is a null externalptr (it may have been disconnected from "
         " the session where it was created)");
  }
}


// [[Rcpp::export]]
bool py_numpy_available_impl() {
  return haveNumPy();
}


// [[Rcpp::export]]
std::vector<std::string> py_list_attributes_impl(PyObjectRef x) {
  std::vector<std::string> attributes;
  PyObjectPtr attrs(PyObject_Dir(x));
  if (attrs.is_null())
    stop(py_fetch_error());

  Py_ssize_t len = PyList_Size(attrs);
  for (Py_ssize_t index = 0; index<len; index++) {
    PyObject* item = PyList_GetItem(attrs, index);
    attributes.push_back(as_std_string(item));
  }

  return attributes;
}

// [[Rcpp::export]]
bool py_has_attr_impl(PyObjectRef x, const std::string& name) {
  if (py_is_null_xptr(x))
    return false;
  else
    return PyObject_HasAttrString(x, name.c_str());
}

namespace {

PyObjectRef py_get_common(PyObject* object,
                          bool convert,
                          bool silent)
{
  // if we have an object, return it
  if (object != NULL)
    return py_ref(object, convert);

  // if we're silent, return new reference to Py_None
  if (silent) {
    Py_IncRef(Py_None);
    return py_ref(Py_None, convert);
  }

  // otherwise, throw an R error
  stop(py_fetch_error());

}

} // end anonymous namespace

// [[Rcpp::export]]
PyObjectRef py_get_attr_impl(PyObjectRef x,
                             const std::string& key,
                             bool silent = false)
{
  PyObject* attr = PyObject_GetAttrString(x, key.c_str());
  return py_get_common(attr, x.convert(), silent);
}

// [[Rcpp::export]]
PyObjectRef py_get_item_impl(PyObjectRef x,
                             RObject key,
                             bool silent = false)
{
  PyObjectPtr py_key(r_to_py(key, x.convert()));
  PyObject* item = PyObject_GetItem(x, py_key);
  return py_get_common(item, x.convert(), silent);
}

// [[Rcpp::export]]
void py_set_attr_impl(PyObjectRef x,
                      const std::string& name,
                      RObject value)
{
  PyObjectPtr converted(r_to_py(value, x.convert()));
  int res = PyObject_SetAttrString(x, name.c_str(), converted);
  if (res != 0)
    stop(py_fetch_error());
}

// [[Rcpp::export]]
void py_del_attr_impl(PyObjectRef x,
                      const std::string& name)
{
  int res = PyObject_SetAttrString(x, name.c_str(), NULL);
  if (res != 0)
    stop(py_fetch_error());
}

// [[Rcpp::export]]
void py_set_item_impl(PyObjectRef x,
                      RObject key,
                      RObject val)
{
  PyObjectPtr py_key(r_to_py(key, x.convert()));
  PyObjectPtr py_val(r_to_py(val, x.convert()));

  int res = PyObject_SetItem(x, py_key, py_val);
  if (res != 0)
    stop(py_fetch_error());
}


// [[Rcpp::export]]
IntegerVector py_get_attr_types_impl(
    PyObjectRef x,
    const std::vector<std::string>& attrs,
    bool resolve_properties)
{
  const int UNKNOWN     =  0;
  const int VECTOR      =  1;
  const int ARRAY       =  2;
  const int LIST        =  4;
  const int ENVIRONMENT =  5;
  const int FUNCTION    =  6;

  PyObjectRef type = py_get_attr_impl(x, "__class__");

  std::size_t n = attrs.size();
  IntegerVector types = no_init(n);
  for (std::size_t i = 0; i < n; i++) {

    // check if this is a property; if so, avoid resolving it unless
    // requested as this could imply running arbitrary Python code
    const std::string& name = attrs[i];
    if (!resolve_properties) {
      PyObjectRef attr = py_get_attr_impl(type, name, true);
      if (PyObject_TypeCheck(attr, PyProperty_Type)) {
        types[i] = UNKNOWN;
        continue;
      }
    }

    PyObjectRef attr = py_get_attr_impl(x, name, true);
    if (attr.get() == Py_None)
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

  // unnamed arguments
  PyObjectPtr pyArgs(PyTuple_New(args.length()));
  if (args.length() > 0) {
    for (R_xlen_t i = 0; i<args.size(); i++) {
      PyObject* arg = r_to_py(args.at(i), x.convert());
      // NOTE: reference to arg is "stolen" by the tuple
      int res = PyTuple_SetItem(pyArgs, i, arg);
      if (res != 0)
        stop(py_fetch_error());
    }
  }

  // named arguments
  PyObjectPtr pyKeywords(PyDict_New());
  if (keywords.length() > 0) {
    CharacterVector names = keywords.names();
    SEXP namesSEXP = names;
    for (R_xlen_t i = 0; i<keywords.length(); i++) {
      const char* name = Rf_translateChar(STRING_ELT(namesSEXP, i));
      PyObjectPtr arg(r_to_py(keywords.at(i), x.convert()));
      int res = PyDict_SetItemString(pyKeywords, name, arg);
      if (res != 0)
        stop(py_fetch_error());
    }
  }

  // call the function
  PyObjectPtr res(PyObject_Call(x, pyArgs, pyKeywords));

  // check for error
  if (res.is_null())
    stop(py_fetch_error());

  // return
  return py_ref(res.detach(), x.convert());
}

// [[Rcpp::export]]
PyObjectRef py_dict_impl(const List& keys, const List& items, bool convert) {

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

  if (!PyDict_Check(dict))
    return py_get_item_impl(dict, key, false);

  PyObjectPtr pyKey(r_to_py(key, dict.convert()));

  // NOTE: returns borrowed reference
  PyObject* item = PyDict_GetItem(dict, pyKey);
  if (item == NULL) {
    Py_IncRef(Py_None);
    return py_ref(Py_None, false);
  }

  Py_IncRef(item);
  return py_ref(item, dict.convert());

}

// [[Rcpp::export]]
void py_dict_set_item(PyObjectRef dict, RObject key, RObject val) {

  if (!PyDict_Check(dict))
    return py_set_item_impl(dict, key, val);

  PyObjectPtr py_key(r_to_py(key, dict.convert()));
  PyObjectPtr py_val(r_to_py(val, dict.convert()));
  PyDict_SetItem(dict, py_key, py_val);

}

// [[Rcpp::export]]
int py_dict_length(PyObjectRef dict) {

  if (!PyDict_Check(dict))
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
      stop(py_fetch_error());
  }

  return keys;

}

} // end anonymous namespace

// [[Rcpp::export]]
PyObjectRef py_dict_get_keys(PyObjectRef dict) {
  PyObject* keys = py_dict_get_keys_impl(dict);
  return py_ref(keys, dict.convert());
}

// [[Rcpp::export]]
CharacterVector py_dict_get_keys_as_str(PyObjectRef dict) {

  // get the dictionary keys
  PyObjectPtr py_keys(py_dict_get_keys_impl(dict));

  // iterate over keys and convert to string
  std::vector<std::string> keys;

  PyObjectPtr it(PyObject_GetIter(py_keys));
  if (it.is_null())
    stop(py_fetch_error());

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
      stop(py_fetch_error());

    keys.push_back(as_utf8_r_string(str));

  }

  if (PyErr_Occurred())
    stop(py_fetch_error());

  return CharacterVector(keys.begin(), keys.end());

}


// [[Rcpp::export]]
PyObjectRef py_tuple(const List& items, bool convert) {

  R_xlen_t n = items.length();
  PyObject* tuple = PyTuple_New(n);
  for (R_xlen_t i = 0; i < n; i++) {
    PyObject* item = r_to_py(items.at(i), convert);
    // NOTE: reference to arg is "stolen" by the tuple
    int res = PyTuple_SetItem(tuple, i, item);
    if (res != 0)
      stop(py_fetch_error());
  }

  return py_ref(tuple, convert);

}

// [[Rcpp::export]]
int py_tuple_length(PyObjectRef tuple) {

  if (!PyTuple_Check(tuple))
    return PyObject_Size(tuple);

  return PyTuple_Size(tuple);

}


// [[Rcpp::export]]
PyObjectRef py_module_import(const std::string& module, bool convert) {

  PyObject* pModule = py_import(module);
  if (pModule == NULL)
    stop(py_fetch_error());

  return py_ref(pModule, convert);

}

// [[Rcpp::export]]
void py_module_proxy_import(PyObjectRef proxy) {
  if (proxy.exists("module")) {
    Rcpp::RObject r_module = proxy.getFromEnvironment("module");
    std::string module = as<std::string>(r_module);
    PyObject* pModule = py_import(module);
    if (pModule == NULL)
      stop(py_fetch_error());
    proxy.set(pModule);
    proxy.remove("module");
  } else {
    stop("Module proxy does not contain module name");
  }
}



// [[Rcpp::export]]
CharacterVector py_list_submodules(const std::string& module) {

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

// Traverse a Python iterator or generator

// [[Rcpp::export]]
List py_iterate(PyObjectRef x, Function f) {

  // List to return
  std::vector<RObject> list;

  // get the iterator
  PyObjectPtr iterator(PyObject_GetIter(x));
  if (iterator.is_null())
    stop(py_fetch_error());

  // loop over it
  while (true) {

    // check next item
    PyObjectPtr item(PyIter_Next(iterator));
    if (item.is_null()) {
      // null return means either iteration is done or
      // that there is an error
      if (PyErr_Occurred())
        stop(py_fetch_error());
      else
        break;
    }

    // call the function
    SEXP param = x.convert()
      ? py_to_r(item, x.convert())
      : py_ref(item.detach(), false);

    list.push_back(f(param));
  }

  // return the list
  List rList(list.size());
  for (size_t i = 0; i < list.size(); i++)
    rList[i] = list[i];
  return rList;
}

// [[Rcpp::export]]
SEXP py_iter_next(PyObjectRef iterator, RObject completed) {

  PyObjectPtr item(PyIter_Next(iterator));
  if (item.is_null()) {

    // null could mean that iteraton is done so we check to
    // ensure that an error actually occrred
    if (PyErr_Occurred())
      stop(py_fetch_error());

    // if there wasn't an error then return the 'completed' sentinel
    return completed;

  } else {

    // return R object
    return iterator.convert()
      ? py_to_r(item, true)
      : py_ref(item.detach(), false);

  }
}


// [[Rcpp::export]]
SEXP py_run_string_impl(const std::string& code,
                        bool local = false,
                        bool convert = true)
{
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
      stop(py_fetch_error());

    // return locals dictionary (detach so we don't decref on scope exit)
    return py_ref(locals.detach(), convert);

  } else {

    // run the requested code
    PyObjectPtr res(PyRun_StringFlags(code.c_str(), Py_file_input, globals, globals, NULL));
    if (res.is_null())
      stop(py_fetch_error());

    // because globals is borrowed, we need to incref here
    Py_IncRef(globals);
    return py_ref(globals, convert);

  }

}


// [[Rcpp::export]]
SEXP py_run_file_impl(const std::string& file,
                      bool local = false,
                      bool convert = true)
{
  // expand path
  Function pathExpand("path.expand");
  std::string expanded = as<std::string>(pathExpand(file));

  // read file
  std::ifstream ifs(expanded.c_str());
  if (!ifs)
    stop("Unable to open file '%s' (does it exist?)", file);
  std::string code((std::istreambuf_iterator<char>(ifs)),
                   (std::istreambuf_iterator<char>()));
  if (ifs.fail())
    stop("Error occurred while reading file '%s'", file);

  // execute
  return py_run_string_impl(code, local, convert);
}

// [[Rcpp::export]]
SEXP py_eval_impl(const std::string& code, bool convert = true) {

  // compile the code
  PyObjectPtr compiledCode;
  if (Py_CompileStringExFlags != NULL)
    compiledCode.assign(Py_CompileStringExFlags(code.c_str(), "reticulate_eval", Py_eval_input, NULL, 0));
  else
    compiledCode.assign(Py_CompileString(code.c_str(), "reticulate_eval", Py_eval_input));


  if (compiledCode.is_null())
    stop(py_fetch_error());

  // execute the code
  PyObject* main = PyImport_AddModule("__main__");
  PyObject* dict = PyModule_GetDict(main);
  PyObjectPtr local_dict(PyDict_New());
  PyObjectPtr res(PyEval_EvalCode(compiledCode, dict, local_dict));
  if (res.is_null())
    stop(py_fetch_error());

  // return (convert to R if requested)
  RObject result = convert
    ? py_to_r(res, convert)
    : py_ref(res.detach(), convert);

  return result;

}

// [[Rcpp::export]]
SEXP py_convert_pandas_series(PyObjectRef series) {

  // extract dtype
  PyObjectPtr dtype(PyObject_GetAttrString(series, "dtype"));
  PyObjectPtr name(PyObject_GetAttrString(dtype, "name"));

  RObject R_obj;

  // special treatment for pd.Categorical
  if (as_std_string(name) == "category") {

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
    //RObject ordered = py_to_r(ordered_, true);

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

    factor.attr("class") = "factor";
    factor.attr("levels") = factor_levels;
    if (PyObject_IsTrue(ordered)) factor.attr("ordered") = true;

    R_obj = factor;

  // special treatment for pd.TimeStamp
  // if available, time zone information will be respected,
  // but values returned to R will be in UTC
  } else if (as_std_string(name) == "datetime64[ns]" ||

    // if a time zone is present, dtype is "object"
    PyObject_HasAttrString(series, "dt")) {

    // pd.Series.items() returns an iterator over (index, value) pairs
    PyObjectPtr items(PyObject_CallMethod(series, "items", NULL));

    std::vector<double> posixct;

    while (true) {

      // get next tuple
      PyObjectPtr tuple(PyIter_Next(items));
      if (tuple.is_null()) {
        if (PyErr_Occurred())
          stop(py_fetch_error());
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

  // default case
  } else {

    PyObjectPtr values(PyObject_GetAttrString(series, "values"));
    R_obj = py_to_r(values, series.convert());

  }

  return R_obj;

}

// [[Rcpp::export]]
SEXP py_convert_pandas_df(PyObjectRef df) {

  // pd.DataFrame.items() returns an iterator over (column name, Series) pairs
  PyObjectPtr items(PyObject_CallMethod(df, "items", NULL));
  if (! (PyObject_HasAttrString(items, "__next__") || PyObject_HasAttrString(items, "next")))
    stop("Cannot iterate over object");

  std::vector<RObject> list;

  while (true) {

    // get next tuple
    PyObjectPtr tuple(PyIter_Next(items));
    if (tuple.is_null()) {
      if (PyErr_Occurred())
        stop(py_fetch_error());
      else
        break;
    }

    // access Series in slot 1
    PyObjectPtr series(PySequence_GetItem(tuple, 1));

    // delegate to py_convert_pandas_series
    PyObjectRef series_ref(series.detach(), df.convert());
    RObject R_obj = py_convert_pandas_series(series_ref);

    list.push_back(R_obj);

  }

  return List(list.begin(), list.end());

}

// [[Rcpp::export]]
PyObjectRef r_convert_dataframe(RObject dataframe, bool convert) {

  Function r_convert_dataframe_column =
    Environment::namespace_env("reticulate")["r_convert_dataframe_column"];

  PyObjectPtr dict(PyDict_New());

  CharacterVector names = dataframe.attr("names");
  for (R_xlen_t i = 0, n = Rf_xlength(dataframe); i < n; i++)
  {
    RObject column = VECTOR_ELT(dataframe, i);

    // ensure name is converted to appropriate encoding
    const char* name = is_python3()
      ? Rf_translateCharUTF8(names[i])
      : Rf_translateChar(names[i]);

    int status = 0;
    if (OBJECT(column) == 0) {
      if (is_convertible_to_numpy(column)) {
        PyObjectPtr value(r_to_py_numpy(column, convert));
        status = PyDict_SetItemString(dict, name, value);
      } else {
        PyObjectPtr value(r_to_py_cpp(column, convert));
        status = PyDict_SetItemString(dict, name, value);
      }
    } else {
      PyObjectRef ref(r_convert_dataframe_column(column, convert));
      status = PyDict_SetItemString(dict, name, ref.get());
    }

    if (status != 0)
      stop(py_fetch_error());
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
    stop(py_fetch_error());

  return py_date.detach();
}

} // end anonymous namespace

// [[Rcpp::export]]
PyObjectRef r_convert_date(DateVector dates, bool convert) {

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
void py_set_interrupt_impl() {
  PyErr_SetInterrupt();
}

// [[Rcpp::export]]
SEXP py_list_length(PyObjectRef x) {
  Py_ssize_t value = PyList_Size(x);
  if (value <= static_cast<Py_ssize_t>(INT_MAX))
    return Rf_ScalarInteger((int) value);
  else
    return Rf_ScalarReal((double) value);
}

// [[Rcpp::export]]
SEXP py_len_impl(PyObjectRef x, SEXP defaultValue) {

  Py_ssize_t value = PyObject_Size(x);
  if (PyErr_Occurred()) {
    if (defaultValue == R_NilValue) {
      stop(py_fetch_error());
    } else {
      PyErr_Clear();
      return defaultValue;
    }
  }

  if (value <= static_cast<Py_ssize_t>(INT_MAX))
    return Rf_ScalarInteger((int) value);
  else
    return Rf_ScalarReal((double) value);

}

// [[Rcpp::export]]
SEXP py_bool(PyObjectRef x) {

  // invoke __bool__ method
  PyObjectPtr result(PyObject_CallMethod(x, "__bool__", NULL));
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return Rf_ScalarLogical(0);
  }

  // check whether it's the True value
  return Rf_ScalarLogical(result == Py_True);

}
