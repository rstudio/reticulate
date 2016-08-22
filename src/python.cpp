#include "tensorflow_types.hpp"

// TODO: Capture error text rather than PyError_Print
// TODO: Capture ... (named and un-named args) and forward to call
// TODO: py_object_convert (convert from Python to R). could be as.character,
//   as.matrix, as.logical, etc. Could also be done automatically or via
//   some sort of dynamic type annotation mechanism
// TODO: consider R6 wrapper (would allow custom $ functions)
// TODO: .DollarNames
// TODO: Globally available import function

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

// helper function to wrap a PyObject in an XPtr
PyObjectPtr py_object_ptr(PyObject* object, bool decref = true) {
  PyObjectPtr ptr(object);
  ptr.attr("class") = "py_object";
  return ptr;
}

//' @export
// [[Rcpp::export]]
void py_run_string(const std::string& code)
{
  ::PyRun_SimpleString(code.c_str());
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
PyObjectPtr py_main_module() {
  return py_object_ptr(::PyImport_AddModule("__main__"), false);
}

//' @export
// [[Rcpp::export]]
PyObjectPtr py_import(const std::string& module) {
  PyObject* pModule = ::PyImport_ImportModule(module.c_str());
  if (pModule == NULL) {
    ::PyErr_Print();
    stop("Unable to import module '%s'", module);
  }
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
  if (attr == NULL) {
    ::PyErr_Print();
    stop("Attribute '%s' not found.", name);
  }
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
  Py_DECREF(args);
  Py_DECREF(keywords);
  if (res == NULL) {
    ::PyErr_Print();
    stop("Error calling python function");
  }
  return py_object_ptr(res);
}
