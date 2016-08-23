#include "tensorflow_types.hpp"

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
void py_object_print(PyObjectPtr pObject) {
  ::PyObject_Print(pObject.get(), stdout, Py_PRINT_RAW);
}

//' @export
// [[Rcpp::export]]
PyObjectPtr py_object_get_attr(PyObjectPtr pObject, const std::string& name) {
  PyObject* attr = ::PyObject_GetAttrString(pObject.get(), name.c_str());
  if (attr == NULL)
    stop(py_fetch_error());

  return py_object_ptr(attr);
}

//' @export
// [[Rcpp::export]]
bool py_object_is_callable(PyObjectPtr pObject) {
  return ::PyCallable_Check(pObject.get()) == 1;
}

//' @export
// [[Rcpp::export]]
PyObjectPtr py_object_call(PyObjectPtr pObject) {
  PyObject *args = PyTuple_New(0);
  PyObject *keywords = ::PyDict_New();
  PyObject* res = ::PyObject_Call(pObject.get(), args, keywords);
  ::Py_DecRef(args);
  ::Py_DecRef(keywords);
  if (res == NULL)
    stop(py_fetch_error());

  return py_object_ptr(res);
}

//' @export
// [[Rcpp::export]]
std::vector<std::string> py_list_attributes(PyObjectPtr pObject) {
  std::vector<std::string> attributes;
  PyObject* attrs = ::PyObject_Dir(pObject);
  if (attrs == NULL)
    stop(py_fetch_error());

  Py_ssize_t len = ::PyList_Size(attrs);
  for (Py_ssize_t index = 0; index<len; index++) {
    PyObject* item = ::PyList_GetItem(attrs, index);
    const char* value = ::PyString_AsString(item);
    attributes.push_back(value);
  }

  return attributes;
}



