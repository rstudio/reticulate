#include <Rcpp.h>
using namespace Rcpp;

#include "python.hpp"

//' @export
// [[Rcpp::export]]
void testExpression(const std::string& expression) {

  PythonInterpreter& python = pythonInterpreter();
  python.execute(expression.c_str());
}

//' @export
// [[Rcpp::export]]
void testFile(const std::string& file) {

  PythonInterpreter& python = pythonInterpreter();
  python.executeFile(file);

}

