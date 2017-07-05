
#include "libpython.h"

#include <Rcpp.h>
using namespace Rcpp;

#include "reticulate_types.h"

#include "event_loop.h"

#include <fstream>

using namespace libpython;

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

// forward declare error handling utility
std::string py_fetch_error();

// wrap an R object in a longer-lived python object "capsule"

SEXP r_object_from_capsule(PyObject* capsule) {
  SEXP object = (SEXP)PyCapsule_GetPointer(capsule, NULL);
  if (object == NULL)
    stop(py_fetch_error());
  return object;
}

void free_r_object_capsule(PyObject* capsule) {
  ::R_ReleaseObject(r_object_from_capsule(capsule));
}

PyObject* r_object_capsule(SEXP object) {
  ::R_PreserveObject(object);
  return PyCapsule_New((void*)object, NULL, free_r_object_capsule);
}


// helper class for ensuring decref of PyObject in the current scope
template<typename T>
class PyPtr {
public:
  // attach on creation, decref on destruction
  PyPtr() : object_(NULL) {}
  explicit PyPtr(T* object) : object_(object) {}
  virtual ~PyPtr() {
    if (object_ != NULL)
      Py_DecRef((PyObject*)object_);
  }

  operator T*() const { return object_; }

  T* get() const { return object_; }

  void assign(T* object) { object_ = object; }

  T* detach() {
    T* object = object_;
    object_ = NULL;
    return object;
  }

  bool is_null() const { return object_ == NULL; }

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
#ifdef _WIN32
  return PyUnicode_AsMBCSString(str);
#else
  return PyUnicode_AsEncodedString(str, "utf-8", "ignore");
#endif
}

std::string as_std_string(PyObject* str) {

  PyObjectPtr pStr;
  if (is_python3() && PyUnicode_Check(str)) {
    // python3 requires that we turn PyUnicode into PyBytes before
    // we call PyBytes_AsStringAndSize (whereas python2 would
    // automatically handle unicode in PyBytes_AsStringAndSize)
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

PyObject* as_python_bytes(Rbyte* bytes, size_t len) {
  if (is_python3())
    return PyBytes_FromStringAndSize((const char*)bytes, len);
  else
    return PyString_FromStringAndSize((const char*)bytes, len);
}

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

  PyObjectPtr pStr;
  if (is_python3()) {
    // python3 requires that we turn PyUnicode into PyBytes before
    // we call PyBytes_AsStringAndSize (whereas python2 would
    // automatically handle unicode in PyBytes_AsStringAndSize)
    str = PyUnicode_AsBytes(str);
    pStr.assign(str);
  }

  char* buffer;
  int res = is_python3() ?
    PyBytes_AsStringAndSize(str, &buffer, NULL) :
    PyString_AsStringAndSize(str, &buffer, NULL);
  if (res == -1) {
    py_fetch_error();
    return true;
  } else {
    return false;
  }
}

bool is_python_str(PyObject* x) {

  if (PyUnicode_Check(x) && !has_null_bytes(x))
    return true;

  // python3 doesn't have PyString_* so mask it out (all strings in
  // python3 will get caught by PyUnicode_Check, we'll ignore
  // PyBytes entirely and let it remain a python object)
  else if (!is_python3() && PyString_Check(x) && !has_null_bytes(x))
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
  PyObjectPtr modulePtr(PyObject_GetAttrString(classPtr, "__module__"));
  PyObjectPtr namePtr(PyObject_GetAttrString(classPtr, "__name__"));
  std::ostringstream ostr;
  std::string module = as_std_string(modulePtr) + ".";
  std::string builtin("__builtin__");
  if (module.find(builtin) == 0)
    module.replace(0, builtin.length(), "python.builtin");
  std::string builtins("builtins");
  if (module.find(builtins) == 0)
    module.replace(0, builtins.length(), "python.builtin");
  ostr << module << as_std_string(namePtr);
  return ostr.str();
}

// wrap a PyObject
PyObjectRef py_ref(PyObject* object, bool convert, const std::string& extraClass = "") {

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
    // class
    PyObjectPtr classPtr(PyObject_GetAttrString(object, "__class__"));
    
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
    
    // add the bases to the R class attribute
    Py_ssize_t len = PyTuple_Size(classes);
    for (Py_ssize_t i = 0; i<len; i++) {
      PyObject* base = PyTuple_GetItem(classes, i); // borrowed
      attrClass.push_back(as_r_class(base));
    }
  }

  // add python.builtin.object if we don't already have it
  if (std::find(attrClass.begin(), attrClass.end(), "python.builtin.object")
                                                      == attrClass.end()) {
    attrClass.push_back("python.builtin.object");
  }
  
  // apply class filter
  Rcpp::Environment pkgEnv = Rcpp::Environment::namespace_env("reticulate");
  Rcpp::Function py_filter_classes = pkgEnv["py_filter_classes"];
  attrClass = as<std::vector<std::string> >(py_filter_classes(attrClass));
  
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



// get a string representing the last python error
std::string py_fetch_error() {

  // clear last error
  s_lastError.clear();
  
  // determine error
  std::string error;
  PyObject *excType , *excValue , *excTraceback;
  PyErr_Fetch(&excType , &excValue , &excTraceback);
  PyErr_NormalizeException(&excType, &excValue, &excTraceback);
  PyObjectPtr pExcType(excType);
  PyObjectPtr pExcValue(excValue);
  PyObjectPtr pExcTraceback(excTraceback);

  if (!pExcType.is_null() || !pExcValue.is_null()) {
    std::ostringstream ostr;
    if (!pExcType.is_null()) {
      PyObjectPtr pStr(PyObject_GetAttrString(pExcType, "__name__"));
      std::string type = as_std_string(pStr);
      
      // check for keyboard interrupt
      if (type == "KeyboardInterrupt")
        throw Rcpp::internal::InterruptedException();
      
      // store in last error
      s_lastError.setType(type);
      
      // print
      ostr << type << ": ";
    }
    if (!pExcValue.is_null()) {
      PyObjectPtr pStr(PyObject_Str(pExcValue));
      std::string value = as_std_string(pStr);
      
      // store in last error
      s_lastError.setValue(value);
      
      // print
      ostr << value;
    }

    // check for traceback      
    if (!pExcTraceback.is_null()) {
      // call into python for traceback 
      PyObjectPtr module(py_import("traceback"));
      if (!module.is_null()) {
        PyObjectPtr func(PyObject_GetAttrString(module, "format_tb"));
        if (!func.is_null()) {
          PyObjectPtr tb(PyObject_CallFunctionObjArgs(func, excTraceback, NULL));
          if (!tb.is_null()) {
            
            // get the traceback
            std::vector<std::string> traceback;
            Py_ssize_t len = PyList_Size(tb);
            for (Py_ssize_t i = 0; i<len; i++)
              traceback.push_back(as_std_string(PyList_GetItem(tb, i)));
            
            // store in last error
            s_lastError.setTraceback(traceback);
            
            // print if enabled
            if (traceback_enabled()) {
              ostr << std::endl << std::endl << "Detailed traceback: " << std::endl;
              size_t len = traceback.size();
              for (size_t i = 0; i<len; i++)
                ostr << traceback[i];
            }
          }
        }
      }
    }
    
    error = ostr.str();
    
    // set error message
    s_lastError.setMessage(error);
    
  } else {
    error = "<unknown error>";
  }

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
    return REALSXP;// [[Rcpp::export]]

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

// convert a tuple to a character vector
CharacterVector py_tuple_to_character(PyObject* tuple) {
  Py_ssize_t len = PyTuple_Size(tuple);
  CharacterVector vec(len);
  for (Py_ssize_t i = 0; i<len; i++)
    vec[i] = as_std_string(PyTuple_GetItem(tuple, i));
  return vec;
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


void set_string_element(SEXP rArray, int i, PyObject* pyStr) {
  std::string str = as_std_string(pyStr);
  cetype_t ce = PyUnicode_Check(pyStr) ? CE_UTF8 : CE_NATIVE;
  SEXP strSEXP = Rf_mkCharCE(str.c_str(), ce);
  SET_STRING_ELT(rArray, i, strSEXP);
}
 
 
bool py_is_callable(PyObject* x) {
  return PyCallable_Check(x) == 1 || PyObject_HasAttrString(x, "__call__");
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
      return CharacterVector::create(as_std_string(x));

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
        vec[i] = as_std_string(PyList_GetItem(x, i));
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
    
    // allocate memory for name and value vectors
    Py_ssize_t size = PyDict_Size(x);
    std::vector<std::string> names(size);
    Rcpp::List list(size);
    
    // iterate over dict
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    Py_ssize_t idx = 0;
    while (PyDict_Next(x, &pos, &key, &value)) {
      PyObjectPtr str(PyObject_Str(key));
      names[idx] = as_std_string(str);
      list[idx] = py_to_r(value, convert);
      idx++;
    }
    list.names() = names;
    return list;
    
  }

  // numpy array
  else if (isPyArray(x)) {

    // get the array
    PyArrayObject* array = (PyArrayObject*)x;

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
    array = (PyArrayObject*)PyArray_CastToType(array, descr, NPY_ARRAY_FARRAY);
    if (array == NULL)
      stop(py_fetch_error());

    // ensure we release it within this scope
    PyObjectPtr ptrArray((PyObject*)array);

    // R array to return
    SEXP rArray = R_NilValue;

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
          for (npy_intp i=0; i<len; i++)
            set_string_element(rArray, i, pData[i]);
          
        // otherwise return a list of objects
        } else {
          rArray = Rf_allocArray(VECSXP, dimsVector);
          RObject protectArray(rArray);
          for (npy_intp i=0; i<len; i++) {
            SEXP data = py_to_r(pData[i], convert);
            SET_VECTOR_ELT(rArray, i, data);
          }
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
      stop("Unsupported array conversion from %d", typenum);
    }
  }

  // iterator/generator
  else if (PyObject_HasAttrString(x, "__iter__") &&
           (PyObject_HasAttrString(x, "next") ||
            PyObject_HasAttrString(x, "__next__"))) {

    // return it raw but add a class so we can create S3 methods for it
    Py_IncRef(x);
    return py_ref(x, true, std::string("python.builtin.iterator"));
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
  
  // default is to return opaque wrapper to python object. we pass convert = true 
  // because if we hit this code then conversion has been either implicitly
  // or explicitly requested.
  else {
    Py_IncRef(x);
    return py_ref(x, true);
  }
}

// convert an R object to a python object (the returned object
// will have an active reference count on it)
PyObject* r_to_py(RObject x, bool convert) {

  int type = x.sexp_type();
  SEXP sexp = x.get__();

  // NULL becomes python None (Py_IncRef since PyTuple_SetItem
  // will steal the passed reference)
  if (x.isNULL()) {
    Py_IncRef(Py_None);
    return Py_None;

  // use py_object attribute if we have it
  } else if (x.hasAttribute("py_object")) {
    PyObjectRef obj = as<PyObjectRef>(x.attr("py_object"));
    Py_IncRef(obj.get());
    return obj.get();    
    
  // pass python objects straight through (Py_IncRef since returning this
  // creates a new reference from the caller)
  } else if (x.inherits("python.builtin.object")) {
    PyObjectRef obj = as<PyObjectRef>(sexp);
    Py_IncRef(obj.get());
    return obj.get();

  // convert arrays and matrixes to numpy (throw error if numpy not available)
  } else if (x.hasAttribute("dim") && requireNumPy()) {

    IntegerVector dimAttrib = x.attr("dim");
    int nd = dimAttrib.length();
    std::vector<npy_intp> dims(nd);
    for (int i = 0; i<nd; i++)
      dims[i] = dimAttrib[i];
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

    // create the matrix
    PyObject* array = PyArray_New(&PyArray_Type,
                                   nd,
                                   &(dims[0]),
                                   typenum,
                                   NULL,
                                   data,
                                   0,
                                   NPY_ARRAY_FARRAY_RO,
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
      PyObjectPtr capsule(r_object_capsule(x));
      
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

  // integer (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == INTSXP) {
    if (LENGTH(sexp) == 1) {
      int value = INTEGER(sexp)[0];
      return PyInt_FromLong(value);
    } else {
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
  } else if (type == REALSXP) {
    if (LENGTH(sexp) == 1) {
      double value = REAL(sexp)[0];
      return PyFloat_FromDouble(value);
    } else {
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
  } else if (type == CPLXSXP) {
    if (LENGTH(sexp) == 1) {
      Rcomplex cplx = COMPLEX(sexp)[0];
      return PyComplex_FromDoubles(cplx.r, cplx.i);
    } else {
      PyObjectPtr list(PyList_New(LENGTH(sexp)));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        Rcomplex cplx = COMPLEX(sexp)[i];
        // NOTE: reference to added value is "stolen" by the list
        int res = PyList_SetItem(list, i, PyComplex_FromDoubles(cplx.r,
                                                                      cplx.i));
        if (res != 0)
          stop(py_fetch_error());
      }
      return list.detach();
    }

  // logical (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == LGLSXP) {
    if (LENGTH(sexp) == 1) {
      int value = LOGICAL(sexp)[0];
      return PyBool_FromLong(value);
    } else {
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
  } else if (type == STRSXP) {
    if (LENGTH(sexp) == 1) {
      return as_python_str(STRING_ELT(sexp, 0));
    } else {
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
  } else if (type == RAWSXP) {

    return as_python_bytes(RAW(sexp), Rf_length(sexp));

  // list
  } else if (type == VECSXP) {
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
    // create a list if there are no names
    } else {
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
  } else if (type == CLOSXP) {

    // create an R object capsule for the R function
    PyObjectPtr capsule(r_object_capsule(x));
    PyCapsule_SetContext(capsule, (void*)convert);

    // check for a py_function_name attribute
    PyObjectPtr pyFunctionName(r_to_py(x.attr("py_function_name"), 
                                       convert));
   
    // create the python wrapper function
    PyObjectPtr module(py_import("rpytools.call"));
    if (module == NULL)
      stop(py_fetch_error());
    PyObjectPtr func(PyObject_GetAttrString(module, "make_python_function"));
    if (func == NULL)
      stop(py_fetch_error());
    PyObjectPtr wrapper(PyObject_CallFunctionObjArgs(func, 
                                                     capsule.get(), 
                                                     pyFunctionName.get(), 
                                                     NULL));
    if (wrapper == NULL)
      stop(py_fetch_error());

    // return the wrapper
    return wrapper.detach();

  } else {
    Rcpp::print(sexp);
    stop("Unable to convert R object to Python type");
  }
}

// [[Rcpp::export]]
PyObjectRef r_to_py_impl(RObject object, bool convert) {
  return py_ref(r_to_py(object, convert), convert);  
}

// custom module used for calling R functions from python wrappers



extern "C" PyObject* call_r_function(PyObject *self, PyObject* args, PyObject* keywords)
{
  // the first argument is always the capsule containing the R function to call
  PyObject* capsule = PyTuple_GetItem(args, 0);
  RObject rFunction = r_object_from_capsule(capsule);
  bool convert = (bool)PyCapsule_GetContext(capsule);

  // convert remainder of positional arguments to R list
  PyObjectPtr funcArgs(PyTuple_GetSlice(args, 1, PyTuple_Size(args)));
  List rArgs;
  if (convert) {
    rArgs = ::py_to_r(funcArgs, convert);
  } else {
    Py_ssize_t len = PyTuple_Size(funcArgs);
    for (Py_ssize_t index = 0; index<len; index++) {
      PyObject* item = PyTuple_GetItem(funcArgs, index);
      Py_IncRef(item);
      rArgs.push_back(py_ref(item, convert));
    }
  }
 
  // get keyword arguments
  List rKeywords;
  if (convert) {
    rKeywords = ::py_to_r(keywords, convert);
  } else {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(keywords, &pos, &key, &value)) {
      PyObjectPtr str(PyObject_Str(key));
      Py_IncRef(value);
      rKeywords[as_std_string(str)] = py_ref(value, convert);
    }
  }

  // combine positional and keyword arguments
  Function append("append");
  rArgs = append(rArgs, rKeywords);

  // call the R function
  Function doCall("do.call");
  RObject result = doCall(rFunction, rArgs);

  // return it's result
  return r_to_py(result, convert);
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

void call_python_function(void* data) {
  
  // cast to call
  PythonCall* call = (PythonCall*)data; 

  // call the function
  PyObjectPtr res(PyObject_CallFunctionObjArgs(call->func, call->data, NULL));
  if (res.is_null()) {
    // don't throw from here as we are in a callback
    std::string msg = py_fetch_error();
    std::cerr << "Error calling event loop task: " << msg << std::endl;
  }
  
  // delete the call object (will decref the members)
  delete call;
}


extern "C" PyObject* register_event_loop_task(PyObject *self, PyObject* args, PyObject* keywords) {
  
  // arguments are the python function to call and an optional data argument
  // capture them and then incref them so they survive past this call (we'll
  // decref them in the call_python_function callback)
  PyObject* func = PyTuple_GetItem(args, 0);
  PyObject* data = PyTuple_GetItem(args, 1);
  
  // create the call object (the func and data will be automaticlaly incref'd then 
  // decrefed when the call object is destroyed)
  PythonCall* call = new PythonCall(func, data);

  // schedule calling the function
  event_loop::register_task(event_loop::Task(call_python_function, (void*)call));
  
  // return none
  Py_IncRef(Py_None);
  return Py_None; 
}


PyMethodDef RPYCallMethods[] = {
  { "call_r_function", (PyCFunction)call_r_function,
    METH_VARARGS | METH_KEYWORDS, "Call an R function" },
  { "register_event_loop_task", (PyCFunction)register_event_loop_task,
    METH_VARARGS | METH_KEYWORDS, "Register an event loop task" },
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
  return PyModule_Create2(&RPYCallModuleDef, _PYTHON3_ABI_VERSION);
}

// forward declare py_run_file
PyObjectRef py_run_file_impl(const std::string& file);

// [[Rcpp::export]]
void py_activate_virtualenv(const std::string& script)
{
  // get main dict
  PyObject* main = PyImport_AddModule("__main__");
  PyObject* mainDict = PyModule_GetDict(main);
  
  // create local dict with __file__
  PyObjectPtr localDict(PyDict_New());
  PyObjectPtr file(as_python_str(script));
  int res = PyDict_SetItemString(localDict, "__file__", file);
  if (res != 0)
    stop(py_fetch_error());
  
  // read the code in the script
  std::ifstream ifs(script.c_str());
  if (!ifs)
    stop("Unable to open file '%s' (does it exist?)", script);
  std::string code((std::istreambuf_iterator<char>(ifs)),
                   (std::istreambuf_iterator<char>()));
  
  // run string
  PyObjectPtr runRes(PyRun_StringFlags(code.c_str(), Py_file_input, mainDict, localDict, NULL));
  if (runRes.is_null())
    stop(py_fetch_error());
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

    // add rpycall module
    PyImport_AppendInittab("rpycall", &initializeRPYCall);

    // initialize python
    Py_Initialize();
    
    const wchar_t *argv[1] = {s_python_v3.c_str()};
    PySys_SetArgv_v3(1, const_cast<wchar_t**>(argv));

  } else {

    // set program name
    s_python = python;
    Py_SetProgramName(const_cast<char*>(s_python.c_str()));

    // set program home
    s_pythonhome = pythonhome;
    Py_SetPythonHome(const_cast<char*>(s_pythonhome.c_str()));

    // initialize python
    Py_Initialize();

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
  
  // poll for events while executing python code
  event_loop::initialize();
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
  PyObjectPtr str(PyObject_Str(x));
  if (str.is_null())
    stop(py_fetch_error());
  return as_std_string(str);
}

// [[Rcpp::export]]
void py_print(PyObjectRef x) {
  PyObjectPtr str(PyObject_Str(x));
  if (str.is_null())
    stop(py_fetch_error());
  Rcout << as_std_string(str) << std::endl;
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

//' Check if a Python object has an attribute
//'
//' @param x Python object
//' @param name Attribute name
//'
//' @return Logical indicating whether it has the specified attribute
//' @export
// [[Rcpp::export]]
bool py_has_attr(PyObjectRef x, const std::string& name) {
  if (py_is_null_xptr(x))
    return false;
  else
    return PyObject_HasAttrString(x, name.c_str());
}


// [[Rcpp::export]]
PyObjectRef py_get_attr_impl(PyObjectRef x, const std::string& name, bool silent = false) {

  PyObject* attr = PyObject_GetAttrString(x, name.c_str());

  if (attr == NULL) {

    std::string err = py_fetch_error();
    
    if (!silent) {
      // error if we aren't silent
      stop(err);
    } else {
      // otherwise set it to PyNone
      attr = Py_None;
      Py_IncRef(attr);
    }
  }

  return py_ref(attr, x.convert());
}

// [[Rcpp::export]]
void py_set_attr_impl(PyObjectRef x, const std::string& name, RObject value) {
  int res = PyObject_SetAttrString(x, name.c_str(), r_to_py(value, x.convert()));
  if (res != 0)
    stop(py_fetch_error());
}
  


// [[Rcpp::export]]
IntegerVector py_get_attribute_types(
    PyObjectRef x,
    const std::vector<std::string>& attributes) {

  const int UNKNOWN     =  0;
  const int VECTOR      =  1;
  const int ARRAY       =  2;
  const int LIST        =  4;
  const int ENVIRONMENT =  5;
  const int FUNCTION    =  6;

  IntegerVector types(attributes.size());
  for (size_t i = 0; i<attributes.size(); i++) {
    PyObjectRef attr = py_get_attr_impl(x, attributes[i], true);
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
  Py_IncRef(res);
  return py_ref(res, x.convert());
}


// [[Rcpp::export]]
PyObjectRef py_dict(const List& keys, const List& items, bool convert) {
  PyObject* dict = PyDict_New();
  for (R_xlen_t i = 0; i<keys.length(); i++) {
    PyObjectPtr key(r_to_py(keys.at(i), convert));
    PyObjectPtr item(r_to_py(items.at(i), convert));
    PyDict_SetItem(dict, key, item);
  }
  return py_ref(dict, convert);
}


// [[Rcpp::export]]
SEXP py_dict_get_item(PyObjectRef dict, RObject key) {
  PyObjectPtr pyKey(r_to_py(key, dict.convert()));
  PyObject* item = PyDict_GetItem(dict, pyKey);
  if (item != NULL) {
    Py_IncRef(item);
    return py_ref(item, dict.convert());
  } else {
    Py_IncRef(Py_None);
    return py_ref(Py_None, false);
  }
}

// [[Rcpp::export]]
void py_dict_set_item(PyObjectRef dict, RObject item, RObject value) {
  PyObjectPtr pyItem(r_to_py(item, dict.convert()));
  PyObjectPtr pyValue(r_to_py(value, dict.convert()));
  PyDict_SetItem(dict, pyItem, pyValue);
}

// [[Rcpp::export]]
int py_dict_length(PyObjectRef dict) {
  return PyDict_Size(dict);
}

// [[Rcpp::export]]
CharacterVector py_dict_get_keys_as_str(PyObjectRef dict) {
    
  // get the keys and check their length
  PyObjectPtr pyKeys(PyDict_Keys(dict));
  Py_ssize_t len = PyList_Size(pyKeys);
  
  // allocate keys to return
  CharacterVector keys(len);

  // get the keys as strings
  for (Py_ssize_t i = 0; i<len; i++) {
    PyObjectPtr str(PyObject_Str(PyList_GetItem(pyKeys, i)));
    if (str.is_null())
      stop(py_fetch_error());
    keys[i] = as_std_string(str);
  }
  
  // return
  return keys;
}


// [[Rcpp::export]]
PyObjectRef py_tuple(const List& items, bool convert) {
  PyObject* tuple = PyTuple_New(items.length());
  for (R_xlen_t i = 0; i<items.length(); i++) {
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
    std::string module = as<std::string>(proxy.getFromEnvironment("module"));
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
    SEXP param = x.convert() ? py_to_r(item, x.convert()) : py_ref(item, false);
    list.push_back(f(param));
  }

  // return the list
  List rList(list.size());
  for (size_t i = 0; i<list.size(); i++)
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
    return iterator.convert() ? py_to_r(item, true) : py_ref(item, false);
  
  }
}


// [[Rcpp::export]]
SEXP py_run_string_impl(const std::string& code, 
                        bool local = false,
                        bool convert = true)
{
  // run string
  PyObject* main = PyImport_AddModule("__main__");
  PyObject* main_dict = PyModule_GetDict(main);
  PyObject* local_dict = NULL;
  PyObjectPtr local_dict_ptr;
  if (local) {
    local_dict_ptr.assign(PyDict_New());
    local_dict = local_dict_ptr.get();
  } else {
    local_dict = main_dict;
  }
  PyObjectPtr res(PyRun_StringFlags(code.c_str(), Py_file_input, 
                                    main_dict, local_dict, NULL));
  if (res.is_null())
    stop(py_fetch_error());

  // return dictionary with objects defined during the execution
  Py_IncRef(local_dict);
  return py_ref(local_dict, convert);
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
  PyObjectPtr compiledCode(Py_CompileString(code.c_str(), "reticulate_eval", Py_eval_input));
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
 Py_IncRef(res);
 if (convert)
   return py_to_r(res, convert);
 else
   return py_ref(res, convert);
}



