
#define LIBPYTHON_CPP
#include "libpython.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

#include <string>
#include <iostream>

namespace {

void lastDLErrorMessage(std::string* pError)
{
#ifdef _WIN32
  LPVOID lpMsgBuf;
  DWORD dw = ::GetLastError();

  DWORD length = ::FormatMessage(
    FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,
      dw,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR) &lpMsgBuf,
      0, NULL );

  if (length != 0)
  {
    std::string msg((LPTSTR)lpMsgBuf);
    LocalFree(lpMsgBuf);
    pError->assign(msg);
  }
  else
  {
    pError->assign("(Unknown error)");
  }
#else
  const char* msg = ::dlerror();
  if (msg != NULL)
    pError->assign(msg);
  else
    pError->assign("(Unknown error)");
#endif
}

bool loadLibrary(const std::string& libPath, void** ppLib, std::string* pError)
{
  *ppLib = NULL;
#ifdef _WIN32
  *ppLib = (void*)::LoadLibraryEx(libPath.c_str(), NULL, 0);
#else
  *ppLib = ::dlopen(libPath.c_str(), RTLD_NOW|RTLD_GLOBAL);
#endif
  if (*ppLib == NULL)
  {
    lastDLErrorMessage(pError);
    *pError = libPath + " - " + *pError;
    return false;
  }
  else
  {
    return true;
  }
}

bool loadSymbol(void* pLib, const std::string& name, void** ppSymbol, std::string* pError)
{
  *ppSymbol = NULL;
#ifdef _WIN32
  *ppSymbol = (void*)::GetProcAddress((HINSTANCE)pLib, name.c_str());
#else
  *ppSymbol = ::dlsym(pLib, name.c_str());
#endif
  if (*ppSymbol == NULL)
  {
    lastDLErrorMessage(pError);
    *pError = name + " - " + *pError;
    return false;
  }
  else
  {
    return true;
  }
}

bool closeLibrary(void* pLib, std::string* pError)
{
#ifdef _WIN32
  if (!::FreeLibrary((HMODULE)pLib))
#else
  if (::dlclose(pLib) != 0)
#endif
  {
    lastDLErrorMessage(pError);
    return false;
  }
  else
  {
    return true;
  }
}

} // anonymous namespace




#define LOAD_PYTHON_SYMBOL_AS(name, as)             \
if (!loadSymbol(pLib_, #name, (void**)&as, pError)) \
  return false;

#define LOAD_PYTHON_SYMBOL(name)                                \
if (!loadSymbol(pLib_, #name, (void**)&_##name, pError)) \
  return false;

bool SharedLibrary::load(const std::string& libPath, bool python3, std::string* pError)
{
  if (!loadLibrary(libPath, &pLib_, pError))
    return false;

  return loadSymbols(python3, pError);
}


bool LibPython::loadSymbols(bool python3, std::string* pError)
{
  bool is64bit = sizeof(size_t) >= 8;

  LOAD_PYTHON_SYMBOL(Py_Initialize)
  LOAD_PYTHON_SYMBOL(Py_IncRef)
  LOAD_PYTHON_SYMBOL(Py_DecRef)
  LOAD_PYTHON_SYMBOL(PyObject_GetAttrString)
  LOAD_PYTHON_SYMBOL(PyObject_HasAttrString)
  LOAD_PYTHON_SYMBOL(PyTuple_Size)
  LOAD_PYTHON_SYMBOL(PyTuple_GetItem)
  LOAD_PYTHON_SYMBOL(PyTuple_New)
  LOAD_PYTHON_SYMBOL(PyTuple_SetItem)
  LOAD_PYTHON_SYMBOL(PyTuple_GetSlice)
  LOAD_PYTHON_SYMBOL(PyList_New)
  LOAD_PYTHON_SYMBOL(PyList_Size)
  LOAD_PYTHON_SYMBOL(PyList_GetItem)
  LOAD_PYTHON_SYMBOL(PyList_SetItem)
  LOAD_PYTHON_SYMBOL(PyErr_Fetch)
  LOAD_PYTHON_SYMBOL(PyErr_Occurred)
  LOAD_PYTHON_SYMBOL(PyErr_NormalizeException)
  LOAD_PYTHON_SYMBOL_AS(PyObject_Str, __PyObject_Str)
  LOAD_PYTHON_SYMBOL(PyObject_Dir)
  LOAD_PYTHON_SYMBOL(PyCallable_Check)
  LOAD_PYTHON_SYMBOL(PyRun_StringFlags)
  LOAD_PYTHON_SYMBOL(PyModule_GetDict)
  LOAD_PYTHON_SYMBOL(PyImport_AddModule)
  LOAD_PYTHON_SYMBOL(PyImport_ImportModule)
  LOAD_PYTHON_SYMBOL(PyObject_GetIter)
  LOAD_PYTHON_SYMBOL(PyIter_Next)
  LOAD_PYTHON_SYMBOL(PyLong_AsLong)
  LOAD_PYTHON_SYMBOL(PyLong_FromLong)
  LOAD_PYTHON_SYMBOL(PyBool_FromLong)
  LOAD_PYTHON_SYMBOL(PyDict_New)
  LOAD_PYTHON_SYMBOL(PyDict_SetItem)
  LOAD_PYTHON_SYMBOL(PyDict_SetItemString)
  LOAD_PYTHON_SYMBOL_AS(PyDict_Next, __PyDict_Next)
  LOAD_PYTHON_SYMBOL(PyFloat_AsDouble)
  LOAD_PYTHON_SYMBOL(PyFloat_FromDouble)
  LOAD_PYTHON_SYMBOL_AS(PyFunction_Type, _PyFunction_Type)
  LOAD_PYTHON_SYMBOL_AS(PyModule_Type, _PyModule_Type)
  LOAD_PYTHON_SYMBOL(PyComplex_FromDoubles)
  LOAD_PYTHON_SYMBOL(PyComplex_RealAsDouble)
  LOAD_PYTHON_SYMBOL(PyComplex_ImagAsDouble)
  LOAD_PYTHON_SYMBOL(PyObject_IsInstance)
  LOAD_PYTHON_SYMBOL(PyObject_Call)
  LOAD_PYTHON_SYMBOL(PyObject_CallFunctionObjArgs)
  LOAD_PYTHON_SYMBOL(PyType_IsSubtype)

  if (python3) {
    LOAD_PYTHON_SYMBOL(PyModule_Create2)
    LOAD_PYTHON_SYMBOL(PyImport_AppendInittab)
    LOAD_PYTHON_SYMBOL_AS(PySys_SetArgv, _PySys_SetArgv_v3)
    LOAD_PYTHON_SYMBOL(PyUnicode_EncodeLocale)
    LOAD_PYTHON_SYMBOL(PyBytes_AsStringAndSize)
    LOAD_PYTHON_SYMBOL(PyBytes_FromStringAndSize)
    LOAD_PYTHON_SYMBOL(PyUnicode_FromString)
    LOAD_PYTHON_SYMBOL_AS(PyLong_AsLong, _PyInt_AsLong)
    LOAD_PYTHON_SYMBOL_AS(PyLong_FromLong, _PyInt_FromLong)
  } else {
    if (is64bit) {
      LOAD_PYTHON_SYMBOL_AS(Py_InitModule4_64, _Py_InitModule4)
    } else {
      LOAD_PYTHON_SYMBOL(Py_InitModule4)
    }
    LOAD_PYTHON_SYMBOL(PyString_AsStringAndSize)
    LOAD_PYTHON_SYMBOL(PyString_FromStringAndSize)
    LOAD_PYTHON_SYMBOL(PyString_FromString)
    LOAD_PYTHON_SYMBOL(PySys_SetArgv)
    LOAD_PYTHON_SYMBOL(PyInt_AsLong)
    LOAD_PYTHON_SYMBOL(PyInt_FromLong)
    LOAD_PYTHON_SYMBOL(PyCObject_AsVoidPtr)
  }
  LOAD_PYTHON_SYMBOL(PyCapsule_New)
  LOAD_PYTHON_SYMBOL(PyCapsule_GetPointer)
  LOAD_PYTHON_SYMBOL(Py_BuildValue)



  _Py_None = _Py_BuildValue("z", NULL);
  _Py_Unicode = _Py_BuildValue("u", "a");
  if (python3)
    _Py_String = _Py_BuildValue("y", "a");
  else
    _Py_String = _Py_BuildValue("s", "a");

  _Py_Int = ::_PyInt_FromLong(1024L);
  _Py_Long = ::_PyLong_FromLong(1024L);
  _Py_Bool = ::_PyBool_FromLong(1L);
  _Py_True = ::_PyBool_FromLong(1L);
  _Py_False = ::_PyBool_FromLong(0L);
  _Py_Dict = ::_Py_BuildValue("{s:i}", "a", 1024);
  _Py_Float = ::_PyFloat_FromDouble(0.0);
  _Py_Tuple = ::_Py_BuildValue("(i)", 1024);
  _Py_List = ::_Py_BuildValue("[i]", 1024);
  _Py_Complex = ::_PyComplex_FromDoubles(0.0, 0.0);

  return true;
}

bool SharedLibrary::unload(std::string* pError)
{
  if (pLib_ != NULL)
    return closeLibrary(pLib_, pError);
  else
    return true;
}





