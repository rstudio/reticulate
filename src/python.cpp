#include <Python.h>
#include <frameobject.h>

#if PY_MAJOR_VERSION >= 3
#define PYTHON_3
// PyInt is gone in python3 so alias to PyLong
#define PyInt_Check PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#define PyInt_FromLong PyLong_FromLong
// Python arguments array becomes whcar_t in python3
#define PYTHON_ARGV L"python"
typedef wchar_t argv_char_t;
#else
#define PYTHON_ARGV "python"
typedef char argv_char_t;
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <dlfcn.h>

#include <Rcpp.h>
using namespace Rcpp;

#include "tensorflow_types.hpp"

// helper class for ensuring decref of PyObject in the current scope
template<typename T>
class PyPtr {
public:
  // attach on creation, decref on destruction
  explicit PyPtr(T* object) : object_(object) {}
  virtual ~PyPtr() {
    if (object_ != NULL)
      Py_DECREF(object_);
  }

  operator T*() const { return object_; }

  T* get() const { return object_; }

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

std::string py_fetch_error();

std::string as_std_string(PyObject* str) {
#ifdef PYTHON_3
  // python3 requires that we turn PyUnicode into PyBytes before
  // we call PyBytes_AsStringAndSize (whereas python2 would
  // automatically handle unicode in PyBytes_AsStringAndSize)
  str = ::PyUnicode_EncodeLocale(str, "strict");
  PyObjectPtr pStr(str);
#endif

  char* buffer;
  Py_ssize_t length;
#ifdef PYTHON_3
  // python3 doesn't have PyString_* APIs, so we use the converted bytes
  if (::PyBytes_AsStringAndSize(str, &buffer, &length) == -1)
#else
  if (::PyString_AsStringAndSize(str, &buffer, &length) == -1)
#endif
    stop(py_fetch_error());
  return std::string(buffer, length);
}

PyObject* as_python_bytes(Rbyte* bytes, size_t len) {
#ifdef PYTHON_3
  return PyBytes_FromStringAndSize((const char*)bytes, len);
#else
  return PyString_FromStringAndSize((const char*)bytes, len);
#endif
}

PyObject* as_python_str(SEXP strSEXP) {
#ifdef PYTHON_3
  // python3 doesn't have PyString and all strings are unicode so
  // make sure we get a unicode representation from R
  const char * value = Rf_translateCharUTF8(strSEXP);
  return PyUnicode_FromString(value);
#else
  const char * value = Rf_translateChar(strSEXP);
  return PyString_FromString(value);
#endif
}

bool has_null_bytes(PyObject* str) {
#ifdef PYTHON_3
  // python3 requires that we turn PyUnicode into PyBytes before
  // we call PyBytes_AsStringAndSize (whereas python2 would
  // automatically handle unicode in PyBytes_AsStringAndSize)
  str = ::PyUnicode_EncodeLocale(str, "strict");
  PyObjectPtr pStr(str);
#endif

  char* buffer;
#ifdef PYTHON_3
  // python3 doesn't have PyString_* APIs, so we use the converted bytes
  if (::PyBytes_AsStringAndSize(str, &buffer, NULL) == -1) {
#else
  if (::PyString_AsStringAndSize(str, &buffer, NULL) == -1) {
#endif
    py_fetch_error();
    return true;
  } else {
    return false;
  }
}

bool is_python_str(PyObject* x) {

  if (PyUnicode_Check(x) && !has_null_bytes(x))
    return true;

#ifndef PYTHON_3
  // python3 doesn't have PyString_* so mask it out (all strings in
  // python3 will get caught by PyUnicode_Check, we'll ignore
  // PyBytes entirely and let it remain a python object)
  else if (PyString_Check(x) && !has_null_bytes(x))
    return true;
#endif

  else
    return false;
}

// check whether a PyObject is None
bool py_is_none(PyObject* object) {
  return object == &::_Py_NoneStruct;
}

std::string as_r_class(PyObject* classPtr) {
  PyObjectPtr modulePtr(::PyObject_GetAttrString(classPtr, "__module__"));
  PyObjectPtr namePtr(::PyObject_GetAttrString(classPtr, "__name__"));
  std::ostringstream ostr;
  std::string module = as_std_string(modulePtr) + ".";
  std::string builtin("__builtin__");
  if (module.find(builtin) == 0)
    module.replace(0, builtin.length(), "tensorflow.builtin");
  std::string builtins("builtins");
  if (module.find(builtins) == 0)
    module.replace(0, builtins.length(), "tensorflow.builtin");
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
  if (::PyObject_HasAttrString(object, "__class__")) {
    // class
    PyObjectPtr classPtr(::PyObject_GetAttrString(object, "__class__"));
    attrClass.push_back(as_r_class(classPtr));

    // base classes
    if (::PyObject_HasAttrString(classPtr, "__bases__")) {
      PyObjectPtr basesPtr(::PyObject_GetAttrString(classPtr, "__bases__"));
      Py_ssize_t len = ::PyTuple_Size(basesPtr);
      for (Py_ssize_t i = 0; i<len; i++) {
        PyObject* base = ::PyTuple_GetItem(basesPtr, i); // borrowed
        attrClass.push_back(as_r_class(base));
      }
    }
  }

  // add tensorflow.builtin.object if we don't already have it
  if (std::find(attrClass.begin(), attrClass.end(), "tensorflow.builtin.object")
                                                      == attrClass.end()) {
    attrClass.push_back("tensorflow.builtin.object");
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
  ::PyErr_Fetch(&excType , &excValue , &excTraceback);
  ::PyErr_NormalizeException(&excType, &excValue, &excTraceback);
  PyObjectPtr pExcType(excType);
  PyObjectPtr pExcValue(excValue);
  PyObjectPtr pExcTraceback(excTraceback);

  if (!pExcType.is_null() || !pExcValue.is_null()) {
    std::ostringstream ostr;
    if (!pExcType.is_null()) {
      PyObjectPtr pStr(::PyObject_GetAttrString(pExcType, "__name__"));
      ostr << as_std_string(pStr) << ": ";
    }
    if (!pExcValue.is_null()) {
      PyObjectPtr pStr(::PyObject_Str(pExcValue));
      ostr << as_std_string(pStr);
    }
    if (!pExcTraceback.is_null()) {
      PyTracebackObject* traceback = (PyTracebackObject*)excTraceback;
      ostr << "\nDetailed traceback: \n";
      while (traceback->tb_next != NULL) {
        traceback = traceback->tb_next;
        PyFrameObject *frame = traceback->tb_frame;
        int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
        std::string filename = as_std_string(frame->f_code->co_filename);
        std::string funcname = as_std_string(frame->f_code->co_name);
        ostr << "    " << filename << " (line" << line << "): " << funcname << "\n";
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

// convert a tuple to a character vector
CharacterVector py_tuple_to_character(PyObject* tuple) {
  Py_ssize_t len = ::PyTuple_Size(tuple);
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
      cplx.r = ::PyComplex_RealAsDouble(x);
      cplx.i = ::PyComplex_ImagAsDouble(x);
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
        cplx.r = ::PyComplex_RealAsDouble(item);
        cplx.i = ::PyComplex_ImagAsDouble(item);
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
  else if (PyTuple_Check(x) && !::PyObject_HasAttrString(x, "_fields")) {
    Py_ssize_t len = ::PyTuple_Size(x);
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

  // iterator/generator
  else if (PyIter_Check(x)) {

    // return it raw but add a class so we can create S3 methods for it
    ::Py_IncRef(x);
    return py_xptr(x, true, "tensorflow.builtin.iterator");
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
    PyArray_DescrPtr descr(PyArray_DescrFromScalar(x));
    int typenum = narrow_array_typenum(descr);
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

  // default is to return opaque wrapper to python object
  else {
    ::Py_IncRef(x);
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
    ::Py_IncRef(&::_Py_NoneStruct);
    return &::_Py_NoneStruct;

  // pass python objects straight through (Py_IncRef since returning this
  // creates a new reference from the caller)
  } else if (x.inherits("tensorflow.builtin.object")) {
    PyObjectXPtr obj = as<PyObjectXPtr>(sexp);
    ::Py_IncRef(obj.get());
    return obj.get();

  // use py_object attribute if we have it
  } else if (x.hasAttribute("py_object")) {
    PyObjectXPtr obj = as<PyObjectXPtr>(x.attr("py_object"));
    ::Py_IncRef(obj.get());
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
    PyObject* matrix = PyArray_New(&PyArray_Type,
                                   nd,
                                   &(dims[0]),
                                   typenum,
                                   NULL,
                                   data,
                                   0,
                                   NPY_ARRAY_FARRAY_RO,
                                   NULL);

    // check for error
    if (matrix == NULL)
      stop(py_fetch_error());

    // return it
    return matrix;

  // integer (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == INTSXP) {
    if (LENGTH(sexp) == 1) {
      int value = INTEGER(sexp)[0];
      return ::PyInt_FromLong(value);
    } else {
      PyObjectPtr list(::PyList_New(LENGTH(sexp)));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        int value = INTEGER(sexp)[i];
        // NOTE: reference to added value is "stolen" by the list
        int res = ::PyList_SetItem(list, i, ::PyInt_FromLong(value));
        if (res != 0)
          stop(py_fetch_error());
      }
      return list.detach();
    }

  // numeric (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == REALSXP) {
    if (LENGTH(sexp) == 1) {
      double value = REAL(sexp)[0];
      return ::PyFloat_FromDouble(value);
    } else {
      PyObjectPtr list(PyList_New(LENGTH(sexp)));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        double value = REAL(sexp)[i];
        // NOTE: reference to added value is "stolen" by the list
        int res = ::PyList_SetItem(list, i, ::PyFloat_FromDouble(value));
        if (res != 0)
          stop(py_fetch_error());
      }
      return list.detach();
    }

  // complex (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == CPLXSXP) {
    if (LENGTH(sexp) == 1) {
      Rcomplex cplx = COMPLEX(sexp)[0];
      return ::PyComplex_FromDoubles(cplx.r, cplx.i);
    } else {
      PyObjectPtr list(PyList_New(LENGTH(sexp)));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        Rcomplex cplx = COMPLEX(sexp)[i];
        // NOTE: reference to added value is "stolen" by the list
        int res = ::PyList_SetItem(list, i, ::PyComplex_FromDoubles(cplx.r,
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
      return ::PyBool_FromLong(value);
    } else {
      PyObjectPtr list(PyList_New(LENGTH(sexp)));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        int value = LOGICAL(sexp)[i];
        // NOTE: reference to added value is "stolen" by the list
        int res = ::PyList_SetItem(list, i, ::PyBool_FromLong(value));
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
      PyObjectPtr list(::PyList_New(LENGTH(sexp)));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        // NOTE: reference to added value is "stolen" by the list
        int res = ::PyList_SetItem(list, i, as_python_str(STRING_ELT(sexp, i)));
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
      PyObjectPtr dict(::PyDict_New());
      CharacterVector names = x.attr("names");
      SEXP namesSEXP = names;
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        const char* name = Rf_translateChar(STRING_ELT(namesSEXP, i));
        PyObjectPtr item(r_to_py(RObject(VECTOR_ELT(sexp, i))));
        int res = ::PyDict_SetItemString(dict, name, item);
        if (res != 0)
          stop(py_fetch_error());
      }
      return dict.detach();
    // create a tuple if there are no names
    } else {
      PyObjectPtr tuple(::PyTuple_New(LENGTH(sexp)));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        PyObject* item = r_to_py(RObject(VECTOR_ELT(sexp, i)));
        // NOTE: reference to added item is "stolen" by tuple
        int res = ::PyTuple_SetItem(tuple, i, item);
        if (res != 0)
          stop(py_fetch_error());
      }
      return tuple.detach();
    }
  } else {
    Rcpp::print(sexp);
    stop("Unable to convert R object to python type");
  }
}


// import numpy array api
bool py_import_numpy_array_api() {
  import_array1(false);
  return true;
}

// [[Rcpp::export]]
void py_initialize(const std::string& pythonSharedLibrary) {

#ifndef __APPLE__
  // force RTLD_GLOBAL of libpython symbols for numpy on Linux, see:
  //
  //   https://bugs.kde.org/show_bug.cgi?id=330032)
  //   http://stackoverflow.com/questions/29880931/
  //
  void * lib = ::dlopen(pythonSharedLibrary.c_str(), RTLD_NOW|RTLD_GLOBAL);
  if (lib == NULL) {
    const char* err = ::dlerror();
    stop(err);
  }
#endif

  // initialize python
  ::Py_Initialize();

  // required to populate sys.
  const argv_char_t *argv[1] = {PYTHON_ARGV};
  PySys_SetArgv(1, const_cast<argv_char_t**>(argv));

  // import numpy array api
  if (!py_import_numpy_array_api())
    stop(py_fetch_error());
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
  return ::PyCallable_Check(x) == 1;
}

// [[Rcpp::export]]
bool py_is_function(PyObjectXPtr x) {
  return PyFunction_Check(x) == 1;
}

// [[Rcpp::export]]
bool py_is_null_xptr(PyObjectXPtr x) {
  return !x;
}


// [[Rcpp::export]]
std::vector<std::string> py_list_attributes(PyObjectXPtr x) {
  std::vector<std::string> attributes;
  PyObjectPtr attrs(::PyObject_Dir(x));
  if (attrs.is_null())
    stop(py_fetch_error());

  Py_ssize_t len = ::PyList_Size(attrs);
  for (Py_ssize_t index = 0; index<len; index++) {
    PyObject* item = ::PyList_GetItem(attrs, index);
    attributes.push_back(as_std_string(item));
  }

  return attributes;
}


// [[Rcpp::export]]
bool py_has_attr(PyObjectXPtr x, const std::string& name) {
  return ::PyObject_HasAttrString(x, name.c_str());
}


// [[Rcpp::export]]
PyObjectXPtr py_get_attr(PyObjectXPtr x, const std::string& name) {
  PyObject* attr = ::PyObject_GetAttrString(x, name.c_str());
  if (attr == NULL)
    stop(py_fetch_error());
  return py_xptr(attr);
}

// [[Rcpp::export]]
IntegerVector py_get_attribute_types(
    PyObjectXPtr x,
    const std::vector<std::string>& attributes) {

  //const int UNKNOWN     =  0;
  const int VECTOR      =  1;
  const int ARRAY       =  2;
  const int LIST        =  4;
  const int ENVIRONMENT =  5;
  const int FUNCTION    =  6;

  IntegerVector types(attributes.size());
  for (size_t i = 0; i<attributes.size(); i++) {
    PyObjectXPtr attr = py_get_attr(x, attributes[i]);
    if (::PyCallable_Check(attr))
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
    else if (PyObject_IsInstance(attr, (PyObject*)&PyModule_Type))
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

// [[Rcpp::export]]
SEXP py_call(PyObjectXPtr x, List args, List keywords = R_NilValue) {

  // unnamed arguments
  PyObjectPtr pyArgs(::PyTuple_New(args.length()));
  for (R_xlen_t i = 0; i<args.size(); i++) {
    PyObject* arg = r_to_py(args.at(i));
    // NOTE: reference to arg is "stolen" by the tuple
    int res = ::PyTuple_SetItem(pyArgs, i, arg);
    if (res != 0)
      stop(py_fetch_error());
  }

  // named arguments
  PyObjectPtr pyKeywords(::PyDict_New());
  if (keywords.length() > 0) {
    CharacterVector names = keywords.names();
    SEXP namesSEXP = names;
    for (R_xlen_t i = 0; i<keywords.length(); i++) {
      const char* name = Rf_translateChar(STRING_ELT(namesSEXP, i));
      PyObjectPtr arg(r_to_py(keywords.at(i)));
      int res = ::PyDict_SetItemString(pyKeywords, name, arg);
      if (res != 0)
        stop(py_fetch_error());
    }
  }

  // call the function
  PyObjectPtr res(::PyObject_Call(x, pyArgs, pyKeywords));

  // check for error
  if (res.is_null())
    stop(py_fetch_error());

  // return as r object
  return py_to_r(res);
}


// [[Rcpp::export]]
PyObjectXPtr py_dict(const List& keys, const List& items) {
  PyObject* dict = ::PyDict_New();
  for (R_xlen_t i = 0; i<keys.length(); i++) {
    PyObjectPtr key(r_to_py(keys.at(i)));
    PyObjectPtr item(r_to_py(items.at(i)));
    ::PyDict_SetItem(dict, key, item);
  }
  return py_xptr(dict);
}

// [[Rcpp::export]]
PyObjectXPtr py_module_impl(const std::string& module) {
  PyObject* pModule = ::PyImport_ImportModule(module.c_str());
  if (pModule == NULL)
    stop(py_fetch_error());
  return py_xptr(pModule);
}


// Traverse a Python iterator or generator

// [[Rcpp::export]]
List py_iterate(PyObjectXPtr x, Function f) {

  // List to return
  List list;

  // get the iterator
  PyObjectPtr iterator(::PyObject_GetIter(x));
  if (iterator.is_null())
    stop(py_fetch_error());

  // loop over it
  while (true) {

    // check next item
    PyObjectPtr item(::PyIter_Next(iterator));
    if (item.is_null()) {
      // null return means either iteration is done or
      // that there is an error
      if (::PyErr_Occurred())
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


// Run Python code
//
// Execute code within the the \code{__main__} Python module.
//
// @param code Code to execute
// @param file File to execute
//
// @name py_run
//
// [[Rcpp::export]]
void py_run_string(const std::string& code)
{
  PyObject* dict = ::PyModule_GetDict(::PyImport_AddModule("__main__"));
  PyObjectPtr res(::PyRun_StringFlags(code.c_str(), Py_file_input, dict, dict, NULL));
  if (res.is_null())
    stop(py_fetch_error());
}


// @rdname py_run
// [[Rcpp::export]]
void py_run_file(const std::string& file)
{
  // expand path
  Function pathExpand("path.expand");
  std::string expanded = as<std::string>(pathExpand(file));

  // open and run
  FILE* fp = ::fopen(expanded.c_str(), "r");
  if (fp) {
    int res = ::PyRun_SimpleFile(fp, expanded.c_str());
    if (res != 0)
      stop(py_fetch_error());
  }
  else
    stop("Unable to read script file '%s' (does the file exist?)", file);
}

