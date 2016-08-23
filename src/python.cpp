#include "tensorflow_types.hpp"

#include <numpy/arrayobject.h>

// TODO: Capture ... (named and un-named args) and forward to call
// TODO: py_object_convert (convert from Python to R). could be as.character,
//   as.matrix, as.logical, etc. Could also be done automatically or via
//   some sort of dynamic type annotation mechanism. we could simply convert
//   anything that we can trivially round-trip back into python

using namespace Rcpp;

// https://docs.python.org/2/c-api/object.html

// [[Rcpp::export]]
void py_initialize() {
  ::Py_Initialize();
  import_array();
}

// [[Rcpp::export]]
void py_finalize() {
  ::Py_Finalize();
}

// wrap a PyObject in an XPtr
PyObjectPtr py_object_ptr(PyObject* object, bool decref = true) {
  PyObjectPtr ptr(object);
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
    ::Py_DecRef(pStr) ;
    ::Py_DecRef(pExcValue);
    return ostr.str();
  } else {
    return "<unknown error>";
  }
}

//' @export
// [[Rcpp::export]]
PyObjectPtr py_main_module() {
  PyObject* main = ::PyImport_AddModule("__main__");
  if (main == NULL)
    stop(py_fetch_error());
  return py_object_ptr(main, false);
}


//' @export
// [[Rcpp::export]]
void py_run_string(const std::string& code)
{
  PyObject* dict = ::PyModule_GetDict(py_main_module());
  PyObject* res  = ::PyRun_StringFlags(code.c_str(), Py_file_input, dict, dict, NULL);
  if (res == NULL)
    stop(py_fetch_error());
  ::Py_DecRef(res);
}

//' @export
// [[Rcpp::export]]
void py_run_file(const std::string& file)
{
  FILE* fp = ::fopen(file.c_str(), "r");
  if (fp)
    ::PyRun_SimpleFile(fp, file.c_str());
  else
    stop("Unable to read script file '%s' (does the file exist?)", file);
}


//' @export
// [[Rcpp::export]]
PyObjectPtr py_import(const std::string& module) {
  PyObject* pModule = ::PyImport_ImportModule(module.c_str());
  if (pModule == NULL)
    stop(py_fetch_error());

  return py_object_ptr(pModule);
}

//' @export
// [[Rcpp::export(print.py_object)]]
void py_object_print(PyObjectPtr x) {
  if (x.get() != (&::_Py_NoneStruct))
    ::PyObject_Print(x, stdout, Py_PRINT_RAW);
}

//' @export
// [[Rcpp::export]]
PyObjectPtr py_object_get_attr(PyObjectPtr x, const std::string& name) {
  PyObject* attr = ::PyObject_GetAttrString(x, name.c_str());
  if (attr == NULL)
    stop(py_fetch_error());

  return py_object_ptr(attr);
}

//' @export
// [[Rcpp::export]]
bool py_object_is_callable(PyObjectPtr x) {
  return ::PyCallable_Check(x) == 1;
}

// convert an R object to a python object (the returned object
// will have an active reference count on it)
PyObject* r_to_py(RObject x) {

  int type = x.sexp_type();
  SEXP sexp = x.get__();

  // NULL and empty vector become python None (Py_IncRef since PyTuple_SetItem
  // will steal the passed reference)
  if (x.isNULL() || (LENGTH(sexp) == 0)) {
    Py_IncRef(Py_None);
    return Py_None;

  // pass python objects straight through (Py_IncRef since PyTuple_SetItem
  // will steal the passed reference)
  } else if (x.inherits("py_object")) {
    PyObjectPtr obj = as<PyObjectPtr>(sexp);
    Py_IncRef(obj.get());
    return obj.get();

  // integer (pass length 1 vectors as scalars, otherwise pass numpy array)
  } else if (type == INTSXP) {
    if (LENGTH(sexp) == 1) {
      int value = INTEGER(sexp)[0];
      return PyInt_FromLong(value);
    } else {
      npy_intp dims = LENGTH(sexp);
      PyObject* array = PyArray_SimpleNewFromData (1, &dims, NPY_INT,
                                                   &(INTEGER(sexp)[0]));
      return array;
    }

  // numeric (pass length 1 vectors as scalars, otherwise pass numpy array)
  } else if (type == REALSXP) {
    if (LENGTH(sexp) == 1) {
      double value = REAL(sexp)[0];
      return PyFloat_FromDouble(value);
    } else {
      npy_intp dims = LENGTH(sexp);
      PyObject* array = PyArray_SimpleNewFromData (1, &dims, NPY_DOUBLE,
                                                   &(REAL(sexp)[0]));
      return array;
    }

  // logical (pass length 1 vectors as scalars, otherwise pass numpy array)
  } else if (type == LGLSXP) {
    if (LENGTH(sexp) == 1) {
      int value = LOGICAL(sexp)[0];
      return PyBool_FromLong(value);
    } else {
      npy_intp dims = LENGTH(sexp);
      PyObject* array = PyArray_SimpleNewFromData (1, &dims, NPY_BOOL,
                                                   &(LOGICAL(sexp)[0]));
      return array;
    }

  // character (pass length 1 vectors as scalars, otherwise pass list)
  } else if (type == STRSXP) {
    if (LENGTH(sexp) == 1) {
      const char* value = CHAR(STRING_ELT(sexp, 0));
      return PyString_FromString(value);
    } else {
      PyObject* list = PyList_New(LENGTH(sexp));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        const char* value = CHAR(STRING_ELT(sexp, i));
        PyList_SetItem(list, i, PyString_FromString(value));
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
          Py_DecRef(dict);
          stop(py_fetch_error());
        }
      }
      return dict;
    // create a list if there are no names
    } else {
      PyObject* list = PyList_New(LENGTH(sexp));
      for (R_xlen_t i = 0; i<LENGTH(sexp); i++) {
        PyObject* item = r_to_py(RObject(VECTOR_ELT(sexp, i)));
        int res = ::PyList_SetItem(list, i, item);
        if (res != 0) {
          Py_DecRef(list);
          stop(py_fetch_error());
        }
      }
      return list;
    }
  } else {
    Rcpp::print(sexp);
    stop("Unable to convert R object to python type");
  }
}


//' @export
// [[Rcpp::export]]
PyObjectPtr py_object_call(PyObjectPtr x, List args, List keywords) {

  // unnamed arguments
  PyObject *pyArgs = ::PyTuple_New(args.length());
  for (R_xlen_t i = 0; i<args.size(); i++) {
    PyObject* arg = r_to_py(args.at(i));
    int res = ::PyTuple_SetItem(pyArgs, i, arg);
    if (res != 0) {
      Py_DecRef(pyArgs);
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
        Py_DecRef(pyKeywords);
        stop(py_fetch_error());
      }
    }
  }

  // call the function
  PyObject* res = ::PyObject_Call(x, pyArgs, pyKeywords);
  ::Py_DecRef(pyArgs);
  ::Py_DecRef(pyKeywords);

  // check for error
  if (res == NULL)
    stop(py_fetch_error());

  // return in PyObject XPtr wrapper
  return py_object_ptr(res);
}

//' @export
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

  ::Py_DecRef(attrs);

  return attributes;
}




