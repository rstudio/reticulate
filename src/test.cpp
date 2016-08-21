#include <Rcpp.h>
using namespace Rcpp;

#include "python.hpp"

//' @export
// [[Rcpp::export]]
void test() {
  python().execute("print('hello');");
}

