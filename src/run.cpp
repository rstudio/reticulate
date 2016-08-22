#include <Rcpp.h>
using namespace Rcpp;

#include "python.hpp"

//' @export
// [[Rcpp::export]]
void py_run_code(const std::string& code) {
  pythonInterpreter().execute(code.c_str());
}

//' @export
// [[Rcpp::export]]
void py_run_file(const std::string& file) {
  pythonInterpreter().executeFile(file);
}


