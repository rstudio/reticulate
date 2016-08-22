#include "tensorflow_types.hpp"

using namespace Rcpp;

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





