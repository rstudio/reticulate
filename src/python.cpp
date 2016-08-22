#include <Rcpp.h>
using namespace Rcpp;

#include "Python.hpp"

#include <boost/make_shared.hpp>

#include <Python.h>

PythonObject::~PythonObject() {
  if (owned_)
    Py_DECREF(pObject_);
}


PythonInterpreter& pythonInterpreter()
{
  static PythonInterpreter instance;
  return instance;
}


PythonInterpreter::PythonInterpreter()
  : mainModule_(::PyImport_AddModule("__main__"), false),
    mainDictionary_(::PyModule_GetDict(mainModule_), false)
{
}


void PythonInterpreter::execute(const std::string& code)
{
  ::PyRun_SimpleString(code.c_str());
}

void PythonInterpreter::executeFile(const std::string& file)
{
  FILE* fp = ::fopen(file.c_str(), "r");
  if (fp)
    ::PyRun_SimpleFile(fp, file.c_str());
  else
    stop("Unable to read script file '%s' (does the file exist?)", file);
}




