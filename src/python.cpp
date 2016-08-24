#include <Python.h>
#include <numpy/arrayobject.h>

#include <Rcpp.h>
using namespace Rcpp;

#include "tensorflow_types.hpp"

// TODO: verify memory management
// TODO: write tests

// TODO: complete marshalling (numpy arrays and matrixes)
// TODO: completion

// check whether a PyObject is None
bool py_is_none(PyObject* object) {
  return object == &::_Py_NoneStruct;
}

// safely decrmement a PyObject
void py_decref(PyObject* object) {
  if (object != NULL)
    ::Py_DecRef(object);
}

// wrap a PyObject in an XPtr
PyObjectPtr py_xptr(PyObject* object, bool decref = true) {
  PyObjectPtr ptr(object, decref);
  ptr.attr("class") = "py_object";
  return ptr;
}

// get a string representing the last python error
std::string py_fetch_error() {
  PyObject *pExcType , *pExcValue , *pExcTraceback;
  ::PyErr_Fetch(&pExcType , &pExcValue , &pExcTraceback) ;
  if (pExcValue != NULL) {
    std::ostringstream ostr;
    PyObject* pStr = ::PyObject_Str(pExcValue) ;
    ostr << ::PyString_AsString(pStr);
    py_decref(pStr) ;
    py_decref(pExcValue);
    return ostr.str();
  } else {
    return "<unknown error>";
  }
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

  // string
  else if (PyString_Check(x))
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
      return wrap(x == Py_True);

    // integer
    else if (scalarType == INTSXP)
      return wrap(PyInt_AsLong(x));

    // double
    else if (scalarType == REALSXP)
      return wrap(PyFloat_AsDouble(x));

    // string
    else if (scalarType == STRSXP)
      return wrap(std::string(PyString_AsString(x)));

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
    } else if (scalarType == LGLSXP) {
      Rcpp::LogicalVector vec(len);
      for (Py_ssize_t i = 0; i<len; i++)
        vec[i] = PyList_GetItem(x, i) == Py_True;
      return vec;
    } else if (scalarType == STRSXP) {
      Rcpp::CharacterVector vec(len);
      for (Py_ssize_t i = 0; i<len; i++)
        vec[i] = PyString_AsString(PyList_GetItem(x, i));
      return vec;
    } else {
      Rcpp::List list(len);
      for (Py_ssize_t i = 0; i<len; i++)
        list[i] = py_to_r(PyList_GetItem(x, i));
      return list;
    }
  }

  // tuple
  else if (PyTuple_Check(x)) {
    Py_ssize_t len = ::PyTuple_Size(x);
    Rcpp::List list(len);
    for (Py_ssize_t i = 0; i<len; i++)
      list[i] = py_to_r(PyTuple_GetItem(x, i));
    return list;
  }

  // dict
  else if (PyDict_Check(x)) {
    // allocate R list
    Py_ssize_t len = ::PyDict_Size(x);
    Rcpp::List list(len);
    // iterate over dict
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(x, &pos, &key, &value))
      list[PyString_AsString(key)] = py_to_r(value);
    return list;
  }

  /*
  // numpy array
  else if (PyArray_Check(x)) {

  }
  */


  // default is to return opaque wrapper for python object
  else
    return py_xptr(x);
}


// convert an R object to a python object (the returned object
// will have an active reference count on it)
PyObject* r_to_py(RObject x) {

  int type = x.sexp_type();
  SEXP sexp = x.get__();
  bool asis = x.inherits("AsIs");

  // NULL and empty vector become python None (Py_IncRef since PyTuple_SetItem
  // will steal the passed reference)
  if (x.isNULL() || (LENGTH(sexp) == 0)) {
    ::Py_IncRef(&::_Py_NoneStruct);
    return &::_Py_NoneStruct;

    // pass python objects straight through (Py_IncRef since PyTuple_SetItem
    // will steal the passed reference)
  } else if (x.inherits("py_object")) {
    PyObjectPtr obj = as<PyObjectPtr>(sexp);
    ::Py_IncRef(obj.get());
    return obj.get();

    // integer (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == INTSXP) {
    if (LENGTH(sexp) == 1 && !asis) {
      int value = INTEGER(sexp)[0];
      return ::PyInt_FromLong(value);
    } else {
      PyObject* list = ::PyList_New(LENGTH(sexp));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        int value = INTEGER(sexp)[i];
        ::PyList_SetItem(list, i, ::PyInt_FromLong(value));
      }
      return list;
    }

    // numeric (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == REALSXP) {
    if (LENGTH(sexp) == 1 && !asis) {
      double value = REAL(sexp)[0];
      return ::PyFloat_FromDouble(value);
    } else {
      PyObject* list = ::PyList_New(LENGTH(sexp));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        double value = REAL(sexp)[i];
        ::PyList_SetItem(list, i, ::PyFloat_FromDouble(value));
      }
      return list;
    }

    // logical (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == LGLSXP) {
    if (LENGTH(sexp) == 1 && !asis) {
      int value = LOGICAL(sexp)[0];
      return ::PyBool_FromLong(value);
    } else {
      PyObject* list = ::PyList_New(LENGTH(sexp));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        int value = LOGICAL(sexp)[i];
        ::PyList_SetItem(list, i, ::PyBool_FromLong(value));
      }
      return list;
    }

    // character (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == STRSXP) {
    if (LENGTH(sexp) == 1 && !asis) {
      const char* value = CHAR(STRING_ELT(sexp, 0));
      return ::PyString_FromString(value);
    } else {
      PyObject* list = ::PyList_New(LENGTH(sexp));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        const char* value = CHAR(STRING_ELT(sexp, i));
        ::PyList_SetItem(list, i, ::PyString_FromString(value));
      }
      return list;
    }

    // list
  } else if (type == VECSXP) {
    // create a dict for names
    if (x.hasAttribute("names")) {
      PyObject *dict = ::PyDict_New();
      CharacterVector names = x.attr("names");
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        const char* name = names.at(i);
        PyObject* item = r_to_py(RObject(VECTOR_ELT(sexp, i)));
        int res = ::PyDict_SetItemString(dict, name, item);
        if (res != 0) {
          py_decref(dict);
          stop(py_fetch_error());
        }
      }
      return dict;
      // create a tuple if there are no names
    } else {
      PyObject* tuple = ::PyTuple_New(LENGTH(sexp));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        PyObject* item = r_to_py(RObject(VECTOR_ELT(sexp, i)));
        int res = ::PyTuple_SetItem(tuple, i, item);
        if (res != 0) {
          py_decref(tuple);
          stop(py_fetch_error());
        }
      }
      return tuple;
    }
  } else {
    Rcpp::print(sexp);
    stop("Unable to convert R object to python type");
  }
}


// [[Rcpp::export]]
void py_initialize() {
  ::Py_Initialize();
  import_array();
}

// [[Rcpp::export]]
void py_finalize() {
  ::Py_Finalize();
}

// [[Rcpp::export]]
bool py_is_none(PyObjectPtr x) {
  return py_is_none(x.get());
}

// [[Rcpp::export]]
void py_print(PyObjectPtr x) {
  PyObject* str = ::PyObject_Str(x);
  if (str == NULL)
    stop(py_fetch_error());
  Rcout << ::PyString_AsString(str);
}

// [[Rcpp::export]]
bool py_is_callable(PyObjectPtr x) {
  return ::PyCallable_Check(x) == 1;
}

// [[Rcpp::export]]
std::vector<std::string> py_list_attributes(PyObjectPtr x) {
  std::vector<std::string> attributes;
  PyObject* attrs = ::PyObject_Dir(x);
  if (attrs == NULL)
    stop(py_fetch_error());

  Py_ssize_t len = ::PyList_Size(attrs);
  for (Py_ssize_t index = 0; index<len; index++) {
    PyObject* item = ::PyList_GetItem(attrs, index);
    const char* value = ::PyString_AsString(item);
    attributes.push_back(value);
  }

  py_decref(attrs);

  return attributes;
}


// [[Rcpp::export]]
PyObjectPtr py_get_attr(PyObjectPtr x, const std::string& name) {
  PyObject* attr = ::PyObject_GetAttrString(x, name.c_str());
  if (attr == NULL)
    stop(py_fetch_error());

  return py_xptr(attr);
}

// [[Rcpp::export]]
SEXP py_to_r(PyObjectPtr x) {
  return py_to_r(x.get());
}

// [[Rcpp::export]]
SEXP py_call(PyObjectPtr x, List args, List keywords) {

  // unnamed arguments
  PyObject *pyArgs = ::PyTuple_New(args.length());
  for (R_xlen_t i = 0; i<args.size(); i++) {
    PyObject* arg = r_to_py(args.at(i));
    int res = ::PyTuple_SetItem(pyArgs, i, arg);
    if (res != 0) {
      py_decref(pyArgs);
      stop(py_fetch_error());
    }
  }

  // named arguments
  PyObject *pyKeywords = ::PyDict_New();
  if (keywords.length() > 0) {
    CharacterVector names = keywords.names();
    for (R_xlen_t i = 0; i<keywords.length(); i++) {
      const char* name = names.at(i);
      PyObject* arg = r_to_py(keywords.at(i));
      int res = ::PyDict_SetItemString(pyKeywords, name, arg);
      if (res != 0) {
        py_decref(pyKeywords);
        stop(py_fetch_error());
      }
    }
  }

  // call the function
  PyObject* res = ::PyObject_Call(x, pyArgs, pyKeywords);
  py_decref(pyArgs);
  py_decref(pyKeywords);

  // check for error
  if (res == NULL)
    stop(py_fetch_error());

  // return R object
  return py_to_r(res);
}



//' Obtain a reference to the main python module
//'
//' @export
// [[Rcpp::export]]
PyObjectPtr py_main_module() {
  PyObject* main = ::PyImport_AddModule("__main__");
  if (main == NULL)
    stop(py_fetch_error());
  return py_xptr(main, false);
}


//' Obtain a reference to a python module
//'
//' @param module Name of module
//'
//' @export
// [[Rcpp::export]]
PyObjectPtr py_import(const std::string& module) {
  PyObject* pModule = ::PyImport_ImportModule(module.c_str());
  if (pModule == NULL)
    stop(py_fetch_error());

  return py_xptr(pModule);
}


//' Run python code
//'
//' @param code Code to run
//'
//' @export
// [[Rcpp::export]]
void py_run_string(const std::string& code)
{
  PyObject* dict = ::PyModule_GetDict(py_main_module());
  PyObject* res  = ::PyRun_StringFlags(code.c_str(), Py_file_input, dict, dict, NULL);
  if (res == NULL)
    stop(py_fetch_error());
  py_decref(res);
}


//' Run python code from a file
//'
//' @param file File to run code from
//'
//' @export
// [[Rcpp::export]]
void py_run_file(const std::string& file)
{
  // expand path
  Function pathExpand("path.expand");
  std::string expanded = as<std::string>(pathExpand(file));

  // open and run
  FILE* fp = ::fopen(expanded.c_str(), "r");
  if (fp)
    ::PyRun_SimpleFile(fp, expanded.c_str());
  else
    stop("Unable to read script file '%s' (does the file exist?)", file);
}




