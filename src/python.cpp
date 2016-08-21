#include <Rcpp.h>
using namespace Rcpp;

#include "Python.hpp"

#include <Python.h>

Python& python()
{
  static Python instance;
  return instance;
}


Python::Python()
{
  ::Py_Initialize();
}


Python::~Python()
{
  try
  {
    ::Py_Finalize();
  }
  catch(...)
  {

  }
}

void Python::execute(const std::string& code)
{
  PyCompilerFlags flags;
  flags.cf_flags = 0;
  ::PyRun_SimpleStringFlags(code.c_str(), &flags);
}

