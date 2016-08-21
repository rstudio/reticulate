
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
