#include <R.h>
#include <Rinternals.h>

#include <cstring>  // for strlen, strchr
#define READLINE_BUFFER_SIZE (8192)
extern "C" int R_ReadConsole(const char*, unsigned char*, int, int);

// [[Rcpp::export]]
SEXP readline(const char* prompt)
{
  // read user input (ensure null termination)
  char buffer[READLINE_BUFFER_SIZE];
  if (R_ReadConsole(prompt, (unsigned char*) buffer, READLINE_BUFFER_SIZE, 1) == 0)
    return R_NilValue;

  buffer[READLINE_BUFFER_SIZE - 1] = '\0';  // Ensure null termination

  // Find the location of the newline character, if any
  char* newline_pos = strchr(buffer, '\n');
  if (newline_pos == nullptr)
    return R_NilValue;  // If no newline found, assume user canceled

  // Determine length up to the newline (excluding the trailing newline)
  size_t input_length = newline_pos - buffer;

  SEXP resultSEXP = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(resultSEXP, 0, Rf_mkCharLen(buffer, input_length));
  UNPROTECT(1);
  return resultSEXP;
}
