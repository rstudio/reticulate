#include <Rcpp.h>
using namespace Rcpp;

#include <R_ext/Print.h>

// [[Rcpp::export]]
int write_stdout(std::string text) {
  Rprintf("%s", text.c_str());
  return text.length();
}


// [[Rcpp::export]]
int write_stderr(std::string text) {
  REprintf("%s", text.c_str());
  return text.length();
}


