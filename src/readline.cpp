#include <Rcpp.h>
using namespace Rcpp;

#include <Rinterface.h>

extern "C" int R_ReadConsole(const char*, unsigned char*, int, int);

// [[Rcpp::export]]
std::string readline(const std::string& prompt)
{
  // read user input (ensure null termination)
  std::size_t n = 8191;
  char buffer[n + 1];
  R_ReadConsole(prompt.c_str(), (unsigned char*) buffer, n, 1);
  buffer[n] = '\0';
  
  // construct resulting string
  std::string result(buffer, buffer + strlen(buffer));
  
  // truncate to location of inserted newline. if not found, assume
  // the buffer overflowed or similar
  std::string::size_type index = result.find('\n');
  if (index == std::string::npos)
    Rf_warning("buffer overflow in readline");
  
  // return result (leaving out trailing newline)
  return result.substr(0, index);
}
