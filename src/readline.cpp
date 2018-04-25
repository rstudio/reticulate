#include <Rcpp.h>
using namespace Rcpp;

#define READLINE_BUFFER_SIZE (8192)
extern "C" int R_ReadConsole(const char*, unsigned char*, int, int);

// [[Rcpp::export]]
SEXP readline(const std::string& prompt)
{
  // read user input (ensure null termination)
  char buffer[READLINE_BUFFER_SIZE];
  R_ReadConsole(prompt.c_str(), (unsigned char*) buffer, READLINE_BUFFER_SIZE, 1);
  buffer[READLINE_BUFFER_SIZE - 1] = '\0';

  // construct resulting string
  std::string result(buffer, buffer + strlen(buffer));

  // truncate to location of inserted newline. if not found, assume
  // the user canceled input with e.g. R_EOF
  std::string::size_type index = result.find('\n');
  if (index == std::string::npos)
    return R_NilValue;

  // return result (leaving out trailing newline)
  SEXP resultSEXP = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(resultSEXP, 0, Rf_mkCharLen(buffer, index));
  UNPROTECT(1);
  return resultSEXP;
}

