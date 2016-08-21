#include <Rcpp.h>
using namespace Rcpp;

#include "python.hpp"

//' @export
// [[Rcpp::export]]
void test() {

  PythonInterpreter& python = pythonInterpreter();
  python.execute("x = 10");


}

