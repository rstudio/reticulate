
#include "libpython.h"

#include <Rcpp.h>
using namespace Rcpp;

#include "reticulate_types.h"

using namespace libpython;

// track whether we are using python 3 (set during py_initialize)
bool s_isPython3 = false;
bool isPython3() {
  return s_isPython3;
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

std::string as_std_string(PyObject* str) {

  PyObjectPtr pStr;
  if (isPython3()) {
    // python3 requires that we turn PyUnicode into PyBytes before
    // we call PyBytes_AsStringAndSize (whereas python2 would
    // automatically handle unicode in PyBytes_AsStringAndSize)
    str = PyUnicode_EncodeLocale(str, "strict");
    pStr.assign(str);
  }

  char* buffer;
  Py_ssize_t length;
  int res = isPython3() ?
    PyBytes_AsStringAndSize(str, &buffer, &length) :
    PyString_AsStringAndSize(str, &buffer, &length);
  if (res == -1)
    stop(py_fetch_error());

  return std::string(buffer, length);
}

PyObject* as_python_bytes(Rbyte* bytes, size_t len) {
  if (isPython3())
    return PyBytes_FromStringAndSize((const char*)bytes, len);
  else
    return PyString_FromStringAndSize((const char*)bytes, len);
}

PyObject* as_python_str(SEXP strSEXP) {
  if (isPython3()) {
    // python3 doesn't have PyString and all strings are unicode so
    // make sure we get a unicode representation from R
    const char * value = Rf_translateCharUTF8(strSEXP);
    return PyUnicode_FromString(value);
  } else {
    const char * value = Rf_translateChar(strSEXP);
    return PyString_FromString(value);
  }
}

bool has_null_bytes(PyObject* str) {

  PyObjectPtr pStr;
  if (isPython3()) {
    // python3 requires that we turn PyUnicode into PyBytes before
    // we call PyBytes_AsStringAndSize (whereas python2 would
    // automatically handle unicode in PyBytes_AsStringAndSize)
    str = PyUnicode_EncodeLocale(str, "strict");
    pStr.assign(str);
  }

  char* buffer;
  int res = isPython3() ?
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
  else if (!isPython3() && PyString_Check(x) && !has_null_bytes(x))
    return true;

  else
    return false;
}

// check whether a PyObject is None
bool py_is_none(PyObject* object) {
  return object == Py_None;
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

// wrap a PyObject in an XPtr
PyObjectXPtr py_xptr(PyObject* object, bool decref = true, const std::string& extraClass = "") {

  // wrap in XPtr
  PyObjectXPtr ptr(object, decref);

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
    attrClass.push_back(as_r_class(classPtr));

    // base classes
    if (PyObject_HasAttrString(classPtr, "__bases__")) {
      PyObjectPtr basesPtr(PyObject_GetAttrString(classPtr, "__bases__"));
      Py_ssize_t len = PyTuple_Size(basesPtr);
      for (Py_ssize_t i = 0; i<len; i++) {
        PyObject* base = PyTuple_GetItem(basesPtr, i); // borrowed
        attrClass.push_back(as_r_class(base));
      }
    }
  }

  // add python.builtin.object if we don't already have it
  if (std::find(attrClass.begin(), attrClass.end(), "python.builtin.object")
                                                      == attrClass.end()) {
    attrClass.push_back("python.builtin.object");
  }

  // add externalptr
  attrClass.push_back("externalptr");

  // set classes
  ptr.attr("class") = attrClass;

  // return XPtr
  return ptr;
}

// get a string representing the last python error
std::string py_fetch_error() {

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
      ostr << as_std_string(pStr) << ": ";
    }
    if (!pExcValue.is_null()) {
      PyObjectPtr pStr(PyObject_Str(pExcValue));
      ostr << as_std_string(pStr);
    }

    if (!pExcTraceback.is_null()) {
      // call into python for traceback printing
      PyObjectPtr module(PyImport_ImportModule("traceback"));
      if (!module.is_null()) {
        PyObjectPtr func(PyObject_GetAttrString(module, "format_tb"));
        if (!func.is_null()) {
          PyObjectPtr tb(PyObject_CallFunctionObjArgs(func, excTraceback, NULL));
          if (!tb.is_null()) {
            ostr << std::endl << std::endl << "Detailed traceback: " << std::endl;
            Py_ssize_t len = PyList_Size(tb);
            for (Py_ssize_t i = 0; i<len; i++)
              ostr << as_std_string(PyList_GetItem(tb, i));
          }
        }
      }
    }
    error = ostr.str();
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
  case NPY_LONG:
  case NPY_LONGLONG:
    typenum = NPY_LONG;
    break;
    // double
  case NPY_FLOAT:
  case NPY_DOUBLE:
    typenum = NPY_DOUBLE;
    break;

    // complex
  case NPY_CFLOAT:
  case NPY_CDOUBLE:
    typenum = NPY_CDOUBLE;
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

// convert a python object to an R object
SEXP py_to_r(PyObject* x) {

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
        list[i] = py_to_r(PyList_GetItem(x, i));
      return list;
    }
  }

  // tuple (but don't convert namedtuple as it's often a custom class)
  else if (PyTuple_Check(x) && !PyObject_HasAttrString(x, "_fields")) {
    Py_ssize_t len = PyTuple_Size(x);
    Rcpp::List list(len);
    for (Py_ssize_t i = 0; i<len; i++)
      list[i] = py_to_r(PyTuple_GetItem(x, i));
    return list;
  }

  // dict
  else if (PyDict_Check(x)) {
    // allocate R list
    Rcpp::List list;
    // iterate over dict
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(x, &pos, &key, &value))
      list[as_std_string(key)] = py_to_r(value);
    return list;
  }

  // numpy array
  else if (PyArray_Check(x)) {

    // get the array
    PyArrayObject* array = (PyArrayObject*)x;

    // get the dimensions
    npy_intp len = PyArray_SIZE(array);
    int nd = PyArray_NDIM(array);
    npy_intp *dims = PyArray_DIMS(array);
    IntegerVector dimsVector(nd);
    for (int i = 0; i<nd; i++)
      dimsVector[i] = dims[i];

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
    }

    // return the R Array
    return rArray;
  }

  // check for numpy scalar
  else if (PyArray_CheckScalar(x)) {

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
    return py_xptr(x, true, "python.builtin.iterator");
  }

  // default is to return opaque wrapper to python object
  else {
    Py_IncRef(x);
    return py_xptr(x);
  }
}

// convert an R object to a python object (the returned object
// will have an active reference count on it)
PyObject* r_to_py(RObject x) {

  int type = x.sexp_type();
  SEXP sexp = x.get__();

  // NULL becomes python None (Py_IncRef since PyTuple_SetItem
  // will steal the passed reference)
  if (x.isNULL()) {
    Py_IncRef(Py_None);
    return Py_None;

  // pass python objects straight through (Py_IncRef since returning this
  // creates a new reference from the caller)
  } else if (x.inherits("python.builtin.object")) {
    PyObjectXPtr obj = as<PyObjectXPtr>(sexp);
    Py_IncRef(obj.get());
    return obj.get();

  // use py_object attribute if we have it
  } else if (x.hasAttribute("py_object")) {
    PyObjectXPtr obj = as<PyObjectXPtr>(x.attr("py_object"));
    Py_IncRef(obj.get());
    return obj.get();

  // convert arrays and matrixes to numpy
  } else if (x.hasAttribute("dim")) {

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
    } else {
      stop("Matrix type cannot be converted to python (only integer, "
           "numeric, complex, and logical matrixes can be converted");
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

    // wrap the R object in a capsule that's tied to the lifetime of the matrix
    // (so the R doesn't deallocate the memory while python is still pointing to it)
    PyObjectPtr capsule(r_object_capsule(x));

    // set the array's base object to the capsule (detach since PyArray_SetBaseObject
    // steals a reference to the provided base object)
    int res = PyArray_SetBaseObject((PyArrayObject *)array, capsule.detach());
    if (res != 0)
      stop(py_fetch_error());

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
        PyObjectPtr item(r_to_py(RObject(VECTOR_ELT(sexp, i))));
        int res = PyDict_SetItemString(dict, name, item);
        if (res != 0)
          stop(py_fetch_error());
      }
      return dict.detach();
    // create a list if there are no names
    } else {
      PyObjectPtr list(PyList_New(LENGTH(sexp)));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        PyObject* item = r_to_py(RObject(VECTOR_ELT(sexp, i)));
        // NOTE: reference to added value is "stolen" by the list
        int res = PyList_SetItem(list, i, item);
        if (res != 0)
          stop(py_fetch_error());
      }
      return list.detach();
    }
  } else if (type == CLOSXP) {

    // create an R object capsule for the R function
    PyObjectPtr capsule(r_object_capsule(sexp));

    // create the python wrapper function
    PyObjectPtr module(PyImport_ImportModule("rpytools.call"));
    if (module == NULL)
      stop(py_fetch_error());
    PyObjectPtr func(PyObject_GetAttrString(module, "make_python_function"));
    if (func == NULL)
      stop(py_fetch_error());
    PyObjectPtr wrapper(PyObject_CallFunctionObjArgs(func, capsule.get(), NULL));
    if (wrapper == NULL)
      stop(py_fetch_error());

    // return the wrapper
    return wrapper.detach();

  } else {
    Rcpp::print(sexp);
    stop("Unable to convert R object to Python type");
  }
}


// custom module used for calling R functions from python wrappers



extern "C" PyObject* call_r_function(PyObject *self, PyObject* args, PyObject* keywords)
{
  // the first argument is always the capsule containing the R function to call
  SEXP rFunction = r_object_from_capsule(PyTuple_GetItem(args, 0));

  // convert remainder of positional arguments to R list
  PyObjectPtr funcArgs(PyTuple_GetSlice(args, 1, PyTuple_Size(args)));
  List rArgs = ::py_to_r(funcArgs);

  // get keyword arguments
  List rKeywords = ::py_to_r(keywords);

  // combine positional and keyword arguments
  Function append("append");
  rArgs = append(rArgs, rKeywords);

  // call the R function
  Function doCall("do.call");
  RObject result = doCall(rFunction, rArgs);

  // return it's result
  return r_to_py(result);
}


PyMethodDef RPYCallMethods[] = {
  { "call_r_function", (PyCFunction)call_r_function,
    METH_VARARGS | METH_KEYWORDS, "Call an R function" },
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
PyObjectXPtr py_run_file(const std::string& file);

// [[Rcpp::export]]
void py_initialize(const std::string& python,
                   const std::string& libpython,
                   const std::string& pythonhome,
                   const std::string& virtualenv_activate,
                   bool python3) {

  // set python3 flag
  s_isPython3 = python3;

  // load the library
  std::string err;
  if (!libPython().load(libpython, isPython3(), &err))
    stop(err);

  if (isPython3()) {

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
  initialize_type_objects(isPython3());

  // execute activate_this.py script for virtualenv if necessary
  if (!virtualenv_activate.empty())
    py_run_file(virtualenv_activate);

  if (!import_numpy_api(isPython3(), &err))
    stop(err);

}

// [[Rcpp::export]]
void py_finalize() {

  // multiple calls to PyFinalize are likely to cause problems so
  // we comment this out to play better with other packages that include
  // python embedding code.

  // ::Py_Finalize();
}

// [[Rcpp::export]]
bool py_is_none(PyObjectXPtr x) {
  return py_is_none(x.get());
}

// [[Rcpp::export]]
CharacterVector py_str(PyObjectXPtr x) {
  PyObjectPtr str(PyObject_Str(x));
  if (str.is_null())
    stop(py_fetch_error());
  return as_std_string(str);
}

// [[Rcpp::export]]
void py_print(PyObjectXPtr x) {
  PyObjectPtr str(PyObject_Str(x));
  if (str.is_null())
    stop(py_fetch_error());
  Rcout << as_std_string(str) << std::endl;
}

// [[Rcpp::export]]
bool py_is_callable(PyObjectXPtr x) {
  return PyCallable_Check(x) == 1;
}

// [[Rcpp::export]]
bool py_is_function(PyObjectXPtr x) {
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
//' @export
// [[Rcpp::export]]
bool py_is_null_xptr(PyObjectXPtr x) {
  return !x;
}


// [[Rcpp::export]]
std::vector<std::string> py_list_attributes(PyObjectXPtr x) {
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
bool py_has_attr(PyObjectXPtr x, const std::string& name) {
  return PyObject_HasAttrString(x, name.c_str());
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
PyObjectXPtr py_get_attr(PyObjectXPtr x, const std::string& name, bool silent = false) {

  PyObject* attr = PyObject_GetAttrString(x, name.c_str());

  if (attr == NULL) {

    if (!silent) {
      // error if we aren't silent
      stop(py_fetch_error());
    } else {
      // otherwise set it to PyNone
      attr = Py_None;
      Py_IncRef(attr);
    }
  }

  return py_xptr(attr);
}

// [[Rcpp::export]]
IntegerVector py_get_attribute_types(
    PyObjectXPtr x,
    const std::vector<std::string>& attributes) {

  const int UNKNOWN     =  0;
  const int VECTOR      =  1;
  const int ARRAY       =  2;
  const int LIST        =  4;
  const int ENVIRONMENT =  5;
  const int FUNCTION    =  6;

  IntegerVector types(attributes.size());
  for (size_t i = 0; i<attributes.size(); i++) {
    PyObjectXPtr attr = py_get_attr(x, attributes[i], true);
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
    else if (PyArray_Check(attr))
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
SEXP py_to_r(PyObjectXPtr x) {
  return py_to_r(x.get());
}


//' Call a Python callable object
//'
//' @param args List of unnamed arguments
//' @param keywords List of named arguments
//'
//' @return Return value of call
//'
//' @keywords internal
//'
//' @export
// [[Rcpp::export]]
SEXP py_call(PyObjectXPtr x, List args, List keywords = R_NilValue) {

  // unnamed arguments
  PyObjectPtr pyArgs(PyTuple_New(args.length()));
  for (R_xlen_t i = 0; i<args.size(); i++) {
    PyObject* arg = r_to_py(args.at(i));
    // NOTE: reference to arg is "stolen" by the tuple
    int res = PyTuple_SetItem(pyArgs, i, arg);
    if (res != 0)
      stop(py_fetch_error());
  }

  // named arguments
  PyObjectPtr pyKeywords(PyDict_New());
  if (keywords.length() > 0) {
    CharacterVector names = keywords.names();
    SEXP namesSEXP = names;
    for (R_xlen_t i = 0; i<keywords.length(); i++) {
      const char* name = Rf_translateChar(STRING_ELT(namesSEXP, i));
      PyObjectPtr arg(r_to_py(keywords.at(i)));
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

  // return as r object
  return py_to_r(res);
}


// [[Rcpp::export]]
PyObjectXPtr py_dict(const List& keys, const List& items) {
  PyObject* dict = PyDict_New();
  for (R_xlen_t i = 0; i<keys.length(); i++) {
    PyObjectPtr key(r_to_py(keys.at(i)));
    PyObjectPtr item(r_to_py(items.at(i)));
    PyDict_SetItem(dict, key, item);
  }
  return py_xptr(dict);
}

// [[Rcpp::export]]
PyObjectXPtr py_tuple(const List& items) {
  PyObject* tuple = PyTuple_New(items.length());
  for (R_xlen_t i = 0; i<items.length(); i++) {
    PyObject* item = r_to_py(items.at(i));
    // NOTE: reference to arg is "stolen" by the tuple
    int res = PyTuple_SetItem(tuple, i, item);
    if (res != 0)
      stop(py_fetch_error());
  }
  return py_xptr(tuple);
}

// [[Rcpp::export]]
PyObjectXPtr py_module_impl(const std::string& module) {
  PyObject* pModule = PyImport_ImportModule(module.c_str());
  if (pModule == NULL)
    stop(py_fetch_error());
  return py_xptr(pModule);
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
List py_iterate(PyObjectXPtr x, Function f) {

  // List to return
  List list;

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

    // call the function and add it's result to the list
    list.push_back(f(py_to_r(item)));
  }

  // return the list
  return list;
}


//' Run Python code
//'
//' Execute code within the the \code{__main__} Python module.
//'
//' @param code Code to execute
//' @param file File to execute
//'
//' @return Reference to \code{__main__} Python module.
//'
//' @name py_run
//'
//' @export
// [[Rcpp::export]]
PyObjectXPtr py_run_string(const std::string& code)
{
  // run string
  PyObject* main = PyImport_AddModule("__main__");
  PyObject* dict = PyModule_GetDict(main);
  PyObjectPtr res(PyRun_StringFlags(code.c_str(), Py_file_input, dict, dict, NULL));
  if (res.is_null())
    stop(py_fetch_error());

  // return reference to main module
  Py_IncRef(main);
  return py_xptr(main);
}


//' @rdname py_run
//' @export
// [[Rcpp::export]]
PyObjectXPtr py_run_file(const std::string& file)
{
  // expand path
  Function pathExpand("path.expand");
  std::string expanded = as<std::string>(pathExpand(file));

  // open and run
  FILE* fp = ::fopen(expanded.c_str(), "r");
  if (fp) {
    int res = PyRun_SimpleFileExFlags(fp, expanded.c_str(), 0, NULL);
    if (res != 0)
      stop(py_fetch_error());

    // return reference to main module
    PyObject* main = PyImport_AddModule("__main__");
    Py_IncRef(main);
    return py_xptr(main);
  }
  else
    stop("Unable to read script file '%s' (does the file exist?)", file);
}

