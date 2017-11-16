
#define LIBPYTHON_CPP
#include "libpython.h"

#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

namespace libpython {

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


bool loadSymbol(void* pLib, const std::vector<std::string>& names, void** ppSymbol, std::string* pError) {
  
  // search for a symbol with one of the specified names
  for (size_t i = 0; i<names.size(); ++i) {
    std::string name = names[i];
    if (loadSymbol(pLib, name, ppSymbol, pError))
      return true;
  }
  
  // return false if nonone found
  return false;
}

} // anonymous namespace



void initialize_type_objects(bool python3) {
  Py_None = Py_BuildValue("z", NULL);
  Py_Unicode = Py_BuildValue("u", L"a");
  if (python3)
    Py_String = Py_BuildValue("y", "a");
  else
    Py_String = Py_BuildValue("s", "a");
  Py_Int = PyInt_FromLong(1024L);
  Py_Long = PyLong_FromLong(1024L);
  Py_Bool = PyBool_FromLong(1L);
  Py_True = PyBool_FromLong(1L);
  Py_False = PyBool_FromLong(0L);
  Py_Dict = Py_BuildValue("{s:i}", "a", 1024);
  Py_Float = PyFloat_FromDouble(0.0);
  Py_Tuple = Py_BuildValue("(i)", 1024);
  Py_List = Py_BuildValue("[i]", 1024);
  Py_Complex = PyComplex_FromDoubles(0.0, 0.0);
}

#define LOAD_PYTHON_SYMBOL_AS(name, as)             \
if (!loadSymbol(pLib_, #name, (void**)&as, pError)) \
  return false;

#define LOAD_PYTHON_SYMBOL(name)                                \
if (!loadSymbol(pLib_, #name, (void**) &libpython::name, pError)) \
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
  LOAD_PYTHON_SYMBOL(Py_IsInitialized)
  LOAD_PYTHON_SYMBOL(Py_AddPendingCall)
  LOAD_PYTHON_SYMBOL(PyErr_SetInterrupt)
  LOAD_PYTHON_SYMBOL(PyExc_KeyboardInterrupt)
  LOAD_PYTHON_SYMBOL(Py_IncRef)
  LOAD_PYTHON_SYMBOL(Py_DecRef)
  LOAD_PYTHON_SYMBOL(PyObject_GetAttrString)
  LOAD_PYTHON_SYMBOL(PyObject_HasAttrString)
  LOAD_PYTHON_SYMBOL(PyObject_SetAttrString)
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
  LOAD_PYTHON_SYMBOL(PyErr_ExceptionMatches)
  LOAD_PYTHON_SYMBOL(PyErr_GivenExceptionMatches)
  LOAD_PYTHON_SYMBOL(PyObject_Str)
  LOAD_PYTHON_SYMBOL(PyObject_Dir)
  LOAD_PYTHON_SYMBOL(PyCallable_Check)
  LOAD_PYTHON_SYMBOL(PyRun_StringFlags)
  LOAD_PYTHON_SYMBOL(Py_CompileString)
  LOAD_PYTHON_SYMBOL(PyEval_EvalCode)
  LOAD_PYTHON_SYMBOL(PyModule_GetDict)
  LOAD_PYTHON_SYMBOL(PyImport_AddModule)
  LOAD_PYTHON_SYMBOL(PyImport_ImportModule)
  LOAD_PYTHON_SYMBOL(PyImport_Import)
  LOAD_PYTHON_SYMBOL(PyImport_GetModuleDict)
  LOAD_PYTHON_SYMBOL(PyObject_GetIter)
  LOAD_PYTHON_SYMBOL(PyIter_Next)
  LOAD_PYTHON_SYMBOL(PyLong_AsLong)
  LOAD_PYTHON_SYMBOL(PyLong_FromLong)
  LOAD_PYTHON_SYMBOL(PyBool_FromLong)
  LOAD_PYTHON_SYMBOL(PyDict_New)
  LOAD_PYTHON_SYMBOL(PyDict_Contains)
  LOAD_PYTHON_SYMBOL(PyDict_GetItem)
  LOAD_PYTHON_SYMBOL(PyDict_SetItem)
  LOAD_PYTHON_SYMBOL(PyDict_SetItemString)
  LOAD_PYTHON_SYMBOL(PyDict_Next)
  LOAD_PYTHON_SYMBOL(PyDict_Keys)
  LOAD_PYTHON_SYMBOL(PyDict_Values)
  LOAD_PYTHON_SYMBOL(PyDict_Size)
  LOAD_PYTHON_SYMBOL(PyDict_Copy)
  LOAD_PYTHON_SYMBOL(PyFloat_AsDouble)
  LOAD_PYTHON_SYMBOL(PyFloat_FromDouble)
  LOAD_PYTHON_SYMBOL(PyFunction_Type)
  LOAD_PYTHON_SYMBOL(PyModule_Type)
  LOAD_PYTHON_SYMBOL(PyType_Type)
  LOAD_PYTHON_SYMBOL(PyComplex_FromDoubles)
  LOAD_PYTHON_SYMBOL(PyComplex_RealAsDouble)
  LOAD_PYTHON_SYMBOL(PyComplex_ImagAsDouble)
  LOAD_PYTHON_SYMBOL(PyObject_IsInstance)
  LOAD_PYTHON_SYMBOL(PyObject_RichCompareBool)
  LOAD_PYTHON_SYMBOL(PyObject_Call)
  LOAD_PYTHON_SYMBOL(PyObject_CallFunctionObjArgs)
  LOAD_PYTHON_SYMBOL(PyType_IsSubtype)
  LOAD_PYTHON_SYMBOL(PySys_WriteStderr)
  LOAD_PYTHON_SYMBOL(PyEval_SetProfile)
  LOAD_PYTHON_SYMBOL(PyGILState_GetThisThreadState)
  LOAD_PYTHON_SYMBOL(PyGILState_Ensure)
  LOAD_PYTHON_SYMBOL(PyGILState_Release)

  // PyUnicode_AsEncodedString may have several different names depending on the Python
  // version and the UCS build type 
  std::vector<std::string> names;
  names.push_back("PyUnicode_AsEncodedString");
  names.push_back("PyUnicodeUCS2_AsEncodedString");
  names.push_back("PyUnicodeUCS4_AsEncodedString");
  if (!loadSymbol(pLib_, names, (void**)&PyUnicode_AsEncodedString, pError) )
    return false;
    
  if (python3) {
    LOAD_PYTHON_SYMBOL(PyModule_Create2)
    LOAD_PYTHON_SYMBOL(PyImport_AppendInittab)
    LOAD_PYTHON_SYMBOL_AS(Py_SetProgramName, Py_SetProgramName_v3)
    LOAD_PYTHON_SYMBOL_AS(Py_SetPythonHome, Py_SetPythonHome_v3)
    LOAD_PYTHON_SYMBOL_AS(PySys_SetArgv, PySys_SetArgv_v3)
    LOAD_PYTHON_SYMBOL(PyUnicode_EncodeLocale)
#ifdef _WIN32
    LOAD_PYTHON_SYMBOL(PyUnicode_AsMBCSString)
#endif
    LOAD_PYTHON_SYMBOL(PyBytes_AsStringAndSize)
    LOAD_PYTHON_SYMBOL(PyBytes_FromStringAndSize)
    LOAD_PYTHON_SYMBOL(PyUnicode_FromString)
    LOAD_PYTHON_SYMBOL_AS(PyLong_AsLong, PyInt_AsLong)
    LOAD_PYTHON_SYMBOL_AS(PyLong_FromLong, PyInt_FromLong)
  } else {
    if (is64bit) {
      LOAD_PYTHON_SYMBOL_AS(Py_InitModule4_64, Py_InitModule4)
    } else {
      LOAD_PYTHON_SYMBOL(Py_InitModule4)
    }
    LOAD_PYTHON_SYMBOL(PyString_AsStringAndSize)
    LOAD_PYTHON_SYMBOL(PyString_FromStringAndSize)
    LOAD_PYTHON_SYMBOL(PyString_FromString)
    LOAD_PYTHON_SYMBOL(Py_SetProgramName)
    LOAD_PYTHON_SYMBOL(Py_SetPythonHome)
    LOAD_PYTHON_SYMBOL(PySys_SetArgv)
    LOAD_PYTHON_SYMBOL(PyInt_AsLong)
    LOAD_PYTHON_SYMBOL(PyInt_FromLong)
    LOAD_PYTHON_SYMBOL(PyCObject_AsVoidPtr)
  }
  LOAD_PYTHON_SYMBOL(PyCapsule_New)
  LOAD_PYTHON_SYMBOL(PyCapsule_GetPointer)
  LOAD_PYTHON_SYMBOL(PyCapsule_SetContext)
  LOAD_PYTHON_SYMBOL(PyCapsule_GetContext)
  LOAD_PYTHON_SYMBOL(Py_BuildValue)

  return true;
}

bool SharedLibrary::unload(std::string* pError)
{
  if (pLib_ != NULL)
    return closeLibrary(pLib_, pError);
  else
    return true;
}

bool import_numpy_api(bool python3, std::string* pError) {
  
  PyObject* numpy = PyImport_ImportModule("numpy.core.multiarray");
  if (numpy == NULL) {
    *pError = "numpy.core.multiarray failed to import";
    return false;
  }
  
  PyObject* c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  Py_DecRef(numpy);
  if (c_api == NULL) {
    *pError = "numpy.core.multiarray _ARRAY_API not found";
    return false;
  }
  
  // get api pointer
  if (python3)
    PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);
  else
    PyArray_API = (void **)PyCObject_AsVoidPtr(c_api);
  
  Py_DecRef(c_api);
  if (PyArray_API == NULL) {
    *pError = "_ARRAY_API is NULL pointer";
    return false;
  }
  
  // check C API version
  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
    std::ostringstream ostr;
    ostr << "incompatible NumPy binary version " << (int) PyArray_GetNDArrayCVersion() << " "
    "(expecting version " << (int) NPY_VERSION << ")";
    *pError = ostr.str();
    return false;
  }
  
  // check feature version 
  if (NPY_1_6_API_VERSION > PyArray_GetNDArrayCFeatureVersion()) {
    std::ostringstream ostr;
    ostr << "incompatible NumPy feature version " << (int) PyArray_GetNDArrayCFeatureVersion() << " "
    "(expecting version " << (int) NPY_1_6_API_VERSION << " or greater)";
    *pError = ostr.str();
    return false;
  }
  
  return true;
}


} // namespace libpython



