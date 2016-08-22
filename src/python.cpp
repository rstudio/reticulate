#include <Python.h>

#include <Rcpp.h>
using namespace Rcpp;

#include "Python.hpp"

#include <boost/make_shared.hpp>

// https://docs.python.org/2/c-api/object.html

PythonObject::~PythonObject() {
  if (owned_ && (pObject_ != NULL))
    Py_DECREF(pObject_);
}


PythonModule::PythonModule(PyObject* module)
  : PythonObject(module, false),
    dictionary_(::PyModule_GetDict(get()), false)
{
}

PythonModule::PythonModule(const char* name)
  : PythonObject(::PyImport_ImportModule(name)),
    dictionary_(::PyModule_GetDict(get()), false)
{
  if (get() == NULL)
    ::PyErr_Print();
}

PythonInterpreter& pythonInterpreter()
{
  static PythonInterpreter instance;
  return instance;
}


PythonInterpreter::PythonInterpreter()
  : pMainModule_(new PythonModule(::PyImport_AddModule("__main__")))
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




