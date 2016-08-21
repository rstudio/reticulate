#include <Rcpp.h>
using namespace Rcpp;

#include "Python.hpp"

#include <boost/make_shared.hpp>

#include <Python.h>

PythonObject::~PythonObject() {
  if (!borrowed_)
    Py_DECREF(pObject_);
}


PythonInterpreter& pythonInterpreter()
{
  static PythonInterpreter instance;
  return instance;
}


PythonInterpreter::PythonInterpreter()
  : mainModule_("__main__")
{
}


void PythonInterpreter::execute(const std::string& code)
{
  PyCompilerFlags flags;
  flags.cf_flags = 0;
  ::PyRun_SimpleStringFlags(code.c_str(), &flags);
}


PythonModule::PythonModule(const char* name)
  : PythonObject(::PyImport_AddModule(name)),
    dictionary_(::PyModule_GetDict(get()), true)
{
}



