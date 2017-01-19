
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

void (*_Py_Initialize)();

_PyObject* (*_Py_InitModule4)(const char *name, _PyMethodDef *methods,
                  const char *doc, _PyObject *self,
                  int apiver);

void (*_Py_IncRef)(_PyObject *);
void (*_Py_DecRef)(_PyObject *);

_PyObject* (*__PyObject_Str)(_PyObject *);

_PyObject* (*_PyObject_Dir)(_PyObject *);

_PyObject* (*_PyObject_GetAttrString)(_PyObject *, const char *);
int (*_PyObject_HasAttrString)(_PyObject*, const char *);

Py_ssize_t (*_PyTuple_Size)(_PyObject *);
_PyObject* (*_PyTuple_GetItem)(_PyObject *, Py_ssize_t);

_PyObject* (*_PyList_New)(Py_ssize_t size);
Py_ssize_t (*_PyList_Size)(_PyObject *);
_PyObject* (*_PyList_GetItem)(_PyObject *, Py_ssize_t);
int (*_PyList_SetItem)(_PyObject *, Py_ssize_t, _PyObject *);

int (*_PyString_AsStringAndSize)(
    register _PyObject *obj,	/* string or Unicode object */
    register char **s,		/* pointer to buffer variable */
    register Py_ssize_t *len	/* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);

_PyObject* (*_PyString_FromString)(const char *);
_PyObject* (*_PyString_FromStringAndSize)(const char *, Py_ssize_t);

void (*_PyErr_Fetch)(_PyObject **, _PyObject **, _PyObject **);
_PyObject* (*_PyErr_Occurred)(void);
void (*_PyErr_NormalizeException)(_PyObject**, _PyObject**, _PyObject**);

int (*_PyCallable_Check)(_PyObject*);

_PyObject* (*_PyModule_GetDict)(_PyObject *);
_PyObject* (*_PyImport_AddModule)(const char *);

_PyObject* (*_PyRun_StringFlags)(const char *, int, _PyObject*, _PyObject*, void*);
int (*_PyRun_SimpleFileExFlags)(FILE *, const char *, int, void *);

_PyObject* (*_PyObject_GetIter)(_PyObject *);
_PyObject* (*_PyIter_Next)(_PyObject *);

void (*_PySys_SetArgv)(int, char **);

_PyObject* (*_PyCapsule_New)(void *pointer, const char *name, _PyCapsule_Destructor destructor);
void* (*_PyCapsule_GetPointer)(_PyObject *capsule, const char *name);

#define LOAD_PYTHON_SYMBOL_AS(name, as)             \
if (!loadSymbol(pLib_, #name, (void**)&as, pError)) \
  return false;

#define LOAD_PYTHON_SYMBOL(name)                                \
if (!loadSymbol(pLib_, #name, (void**)&_##name, pError)) \
  return false;

bool LibPython::load(const std::string& libPath, bool python3, std::string* pError)
{
  if (!loadLibrary(libPath, &pLib_, pError))
    return false;

  bool is64bit = sizeof(size_t) >= 8;

  LOAD_PYTHON_SYMBOL(Py_Initialize)
  LOAD_PYTHON_SYMBOL(Py_IncRef)
  LOAD_PYTHON_SYMBOL(Py_DecRef)
  LOAD_PYTHON_SYMBOL(PyObject_GetAttrString)
  LOAD_PYTHON_SYMBOL(PyObject_HasAttrString)
  LOAD_PYTHON_SYMBOL(PyTuple_Size)
  LOAD_PYTHON_SYMBOL(PyTuple_GetItem)
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
  LOAD_PYTHON_SYMBOL(PyObject_GetIter)
  LOAD_PYTHON_SYMBOL(PyIter_Next)
  LOAD_PYTHON_SYMBOL(PySys_SetArgv)
  if (python3) {

  } else {
    if (is64bit) {
      LOAD_PYTHON_SYMBOL_AS(Py_InitModule4_64, _Py_InitModule4)
    } else {
      LOAD_PYTHON_SYMBOL(Py_InitModule4)
    }
    LOAD_PYTHON_SYMBOL(PyString_AsStringAndSize)
    LOAD_PYTHON_SYMBOL(PyString_FromStringAndSize)
    LOAD_PYTHON_SYMBOL(PyString_FromString)
  }
  LOAD_PYTHON_SYMBOL(PyCapsule_New)
  LOAD_PYTHON_SYMBOL(PyCapsule_GetPointer)


  return true;
}

bool LibPython::unload(std::string* pError)
{
  if (pLib_ != NULL)
    return closeLibrary(pLib_, pError);
  else
    return true;
}





