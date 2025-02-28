
#define LIBPYTHON_CPP
#include "libpython.h"

#ifndef _WIN32
# include <dlfcn.h>
#else
# define WIN32_LEAN_AND_MEAN 1
# include <windows.h>
#endif

#include <R.h>
#include <Rinternals.h>


#include <string>
#include <vector>
#include <iostream>
#include <sstream>

namespace reticulate {
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
  if (libPath == "NA") {
    *ppLib = ::dlopen(NULL, RTLD_NOW|RTLD_GLOBAL); // on linux, should we also do: | RTLD_DEEPBIND ??
  } else {
    *ppLib = ::dlopen(libPath.c_str(), RTLD_NOW|RTLD_GLOBAL);
  }
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
    if (pError != NULL) {
      lastDLErrorMessage(pError);
      *pError = name + " - " + *pError;
    }
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
  Py_ByteArray = PyByteArray_FromStringAndSize("a", 1);
  Py_DictClass = PyObject_Type(Py_Dict);

  PyObject* builtins = PyImport_AddModule(python3 ? "builtins" : "__builtin__"); // borrowed ref
  if (builtins == NULL) goto error;
  PyExc_KeyboardInterrupt = PyObject_GetAttrString(builtins, "KeyboardInterrupt"); // new ref
  PyExc_RuntimeError = PyObject_GetAttrString(builtins, "RuntimeError"); // new ref
  PyExc_AttributeError = PyObject_GetAttrString(builtins, "AttributeError"); // new ref

  if (PyErr_Occurred()) { error:
     // Should never happen. If you see this please report a bug.
     PyErr_Print();
  }
}

#define LOAD_PYTHON_SYMBOL_AS(name, as)             \
if (!loadSymbol(pLib_, #name, (void**)&as, pError)) \
  return false;

#define LOAD_PYTHON_SYMBOL(name)                                \
if (!loadSymbol(pLib_, #name, (void**) &libpython::name, pError)) \
  return false;

bool SharedLibrary::load(const std::string& libPath, int major_ver, int minor_ver, std::string* pError)
{
  if (!loadLibrary(libPath, &pLib_, pError))
    return false;

  return loadSymbols(major_ver, minor_ver, pError);
}

// Define "slow" fallback implementation for Py version <= 3.9
int _PyIter_Check(PyObject* o) {
  return PyObject_HasAttrString(o, "__next__");
}

int _PyObject_GetOptionalAttrString(PyObject* obj, const char* attr_name, PyObject** result) {
  *result = PyObject_GetAttrString(obj, attr_name);
  if (*result == NULL) {
    if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
      PyErr_Clear();
      return 0;
    }
    return -1;
  }
  return 1;
}


bool LibPython::loadSymbols(int python_major_ver, int python_minor_ver, std::string* pError)
{
  bool is64bit = sizeof(size_t) >= 8;

  LOAD_PYTHON_SYMBOL(Py_InitializeEx)
  LOAD_PYTHON_SYMBOL(Py_Finalize)
  LOAD_PYTHON_SYMBOL(Py_IsInitialized)
  LOAD_PYTHON_SYMBOL(Py_GetVersion)          // Deprecated in 3.13
  LOAD_PYTHON_SYMBOL(Py_AddPendingCall)
  LOAD_PYTHON_SYMBOL(Py_MakePendingCalls)
  LOAD_PYTHON_SYMBOL(PyErr_SetInterrupt)
  LOAD_PYTHON_SYMBOL(PyErr_CheckSignals)
  LOAD_PYTHON_SYMBOL(Py_IncRef)
  LOAD_PYTHON_SYMBOL(Py_DecRef)
  LOAD_PYTHON_SYMBOL(PyObject_Size)
  LOAD_PYTHON_SYMBOL(PyObject_Type)
  LOAD_PYTHON_SYMBOL(PyObject_GetAttr)
  LOAD_PYTHON_SYMBOL(PyObject_HasAttr)
  LOAD_PYTHON_SYMBOL(PyObject_SetAttr)
  LOAD_PYTHON_SYMBOL(PyObject_GetAttrString)
  LOAD_PYTHON_SYMBOL(PyObject_HasAttrString)
  if (python_major_ver >= 3 && python_minor_ver >= 13) {
    LOAD_PYTHON_SYMBOL(PyObject_HasAttrStringWithError)
    LOAD_PYTHON_SYMBOL(PyObject_GetOptionalAttrString)
  } else {
    LOAD_PYTHON_SYMBOL_AS(PyObject_HasAttrString, PyObject_HasAttrStringWithError)
    PyObject_GetOptionalAttrString = &_PyObject_GetOptionalAttrString;
  }
  LOAD_PYTHON_SYMBOL(PyObject_SetAttrString)
  LOAD_PYTHON_SYMBOL(PyObject_GetItem)
  LOAD_PYTHON_SYMBOL(PyObject_SetItem)
  LOAD_PYTHON_SYMBOL(PyObject_DelItem)
  LOAD_PYTHON_SYMBOL(PyTuple_Size)
  LOAD_PYTHON_SYMBOL(PyTuple_GetItem)
  LOAD_PYTHON_SYMBOL(PyTuple_New)
  LOAD_PYTHON_SYMBOL(PyTuple_SetItem)
  LOAD_PYTHON_SYMBOL(PyTuple_GetSlice)
  LOAD_PYTHON_SYMBOL(PyList_New)
  LOAD_PYTHON_SYMBOL(PyList_Size)
  LOAD_PYTHON_SYMBOL(PyList_GetItem)
  LOAD_PYTHON_SYMBOL(PyList_SetItem)
  LOAD_PYTHON_SYMBOL(PyErr_Clear)
  LOAD_PYTHON_SYMBOL(PyErr_Print)
  LOAD_PYTHON_SYMBOL(PyErr_Fetch)
  LOAD_PYTHON_SYMBOL(PyErr_Restore)
  LOAD_PYTHON_SYMBOL(PyErr_Occurred)
  LOAD_PYTHON_SYMBOL(PyErr_SetNone)
  LOAD_PYTHON_SYMBOL(PyErr_SetString)
  LOAD_PYTHON_SYMBOL(PyErr_SetObject)
  LOAD_PYTHON_SYMBOL(PyErr_BadArgument)
  LOAD_PYTHON_SYMBOL(PyErr_NormalizeException)
  LOAD_PYTHON_SYMBOL(PyErr_ExceptionMatches)
  LOAD_PYTHON_SYMBOL(PyErr_GivenExceptionMatches)
  LOAD_PYTHON_SYMBOL(PyErr_PrintEx)
  LOAD_PYTHON_SYMBOL(PyObject_Print)
  LOAD_PYTHON_SYMBOL(PyObject_Str)
  LOAD_PYTHON_SYMBOL(PyObject_Repr)
  LOAD_PYTHON_SYMBOL(PyObject_Dir)
  LOAD_PYTHON_SYMBOL(PyByteArray_Size)
  LOAD_PYTHON_SYMBOL(PyByteArray_FromStringAndSize)
  LOAD_PYTHON_SYMBOL(PyByteArray_AsString)
  LOAD_PYTHON_SYMBOL(PyCallable_Check)
  LOAD_PYTHON_SYMBOL(PyRun_StringFlags)
  LOAD_PYTHON_SYMBOL(PyRun_FileEx)
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
  LOAD_PYTHON_SYMBOL(PySlice_New)
  LOAD_PYTHON_SYMBOL(PyBool_FromLong)
  LOAD_PYTHON_SYMBOL(PyDict_New)
  LOAD_PYTHON_SYMBOL(PyDict_Contains)
  LOAD_PYTHON_SYMBOL(PyDict_GetItem)
  LOAD_PYTHON_SYMBOL(PyDict_SetItem)
  LOAD_PYTHON_SYMBOL(PyDict_SetItemString)
  LOAD_PYTHON_SYMBOL(PyDict_DelItemString)
  LOAD_PYTHON_SYMBOL(PyDict_Next)
  LOAD_PYTHON_SYMBOL(PyDict_Keys)
  LOAD_PYTHON_SYMBOL(PyDict_Values)
  LOAD_PYTHON_SYMBOL(PyDict_Size)
  LOAD_PYTHON_SYMBOL(PyDict_Copy)
  LOAD_PYTHON_SYMBOL(PyFloat_AsDouble)
  LOAD_PYTHON_SYMBOL(PyFloat_FromDouble)
  LOAD_PYTHON_SYMBOL(PyFunction_Type)
  LOAD_PYTHON_SYMBOL(PyMethod_Type)
  LOAD_PYTHON_SYMBOL(PyModule_Type)
  LOAD_PYTHON_SYMBOL(PyType_Type)
  LOAD_PYTHON_SYMBOL(PyProperty_Type)
  LOAD_PYTHON_SYMBOL(PyCapsule_IsValid)
  LOAD_PYTHON_SYMBOL(PyComplex_FromDoubles)
  LOAD_PYTHON_SYMBOL(PyComplex_RealAsDouble)
  LOAD_PYTHON_SYMBOL(PyComplex_ImagAsDouble)
  LOAD_PYTHON_SYMBOL(PyObject_IsInstance)
  LOAD_PYTHON_SYMBOL(PyObject_RichCompareBool)
  LOAD_PYTHON_SYMBOL(PyObject_Call)
  LOAD_PYTHON_SYMBOL(PyObject_CallFunctionObjArgs)
  LOAD_PYTHON_SYMBOL(PyType_IsSubtype)
  LOAD_PYTHON_SYMBOL(PyType_GetFlags)
  LOAD_PYTHON_SYMBOL(PyMapping_Items)
  LOAD_PYTHON_SYMBOL(PyOS_getsig)
  LOAD_PYTHON_SYMBOL(PyOS_setsig)
  LOAD_PYTHON_SYMBOL(PySys_WriteStderr)
  LOAD_PYTHON_SYMBOL(PySys_GetObject)
  LOAD_PYTHON_SYMBOL(PyEval_SetProfile)
  LOAD_PYTHON_SYMBOL(PyGILState_GetThisThreadState)
  LOAD_PYTHON_SYMBOL(PyGILState_Ensure)
  LOAD_PYTHON_SYMBOL(PyGILState_Release)
  LOAD_PYTHON_SYMBOL(PyThreadState_Next)
  LOAD_PYTHON_SYMBOL(PyEval_SaveThread)
  LOAD_PYTHON_SYMBOL(PyEval_RestoreThread)
  LOAD_PYTHON_SYMBOL(PyObject_CallMethod)
  LOAD_PYTHON_SYMBOL(PySequence_GetItem)
  LOAD_PYTHON_SYMBOL(PyObject_IsTrue)
  LOAD_PYTHON_SYMBOL(PyCapsule_Import)
  LOAD_PYTHON_SYMBOL(PyUnicode_AsUTF8)
  LOAD_PYTHON_SYMBOL(PyUnicode_CompareWithASCIIString)

  // PyUnicode_AsEncodedString may have several different names depending on the Python
  // version and the UCS build type
  std::vector<std::string> names;
  names.reserve(3);
  names.push_back("PyUnicode_AsEncodedString");
  names.push_back("PyUnicodeUCS2_AsEncodedString");
  names.push_back("PyUnicodeUCS4_AsEncodedString");
  if (!loadSymbol(pLib_, names, (void**)&PyUnicode_AsEncodedString, pError) )
    return false;

  if (python_major_ver >= 3) {
    LOAD_PYTHON_SYMBOL(PyException_SetTraceback)
    LOAD_PYTHON_SYMBOL(Py_GetProgramFullPath)   // Deprecated in 3.13

    // Debug versions of Python will provide PyModule_Create2TraceRefs,
    // while release versions will provide PyModule_Create
#ifdef RETICULATE_PYTHON_DEBUG
    LOAD_PYTHON_SYMBOL_AS(PyModule_Create2TraceRefs, PyModule_Create)
#else
    LOAD_PYTHON_SYMBOL_AS(PyModule_Create2, PyModule_Create)
#endif

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
    LOAD_PYTHON_SYMBOL(Py_CompileStringExFlags)
  } else {
    if (is64bit) {
      LOAD_PYTHON_SYMBOL_AS(Py_InitModule4_64, Py_InitModule4)
    } else {
      LOAD_PYTHON_SYMBOL(Py_InitModule4)
    }
    LOAD_PYTHON_SYMBOL_AS(Py_GetProgramFullPath, Py_GetProgramFullPath_v2)  // Deprecated in 3.13
    LOAD_PYTHON_SYMBOL(PyString_AsStringAndSize)
    LOAD_PYTHON_SYMBOL(PyString_FromStringAndSize)
    LOAD_PYTHON_SYMBOL(PyString_FromString)
    LOAD_PYTHON_SYMBOL(Py_SetProgramName)
    LOAD_PYTHON_SYMBOL(Py_SetPythonHome)
    LOAD_PYTHON_SYMBOL(PySys_SetArgv)
    LOAD_PYTHON_SYMBOL(PyInt_AsLong)
    LOAD_PYTHON_SYMBOL(PyInt_FromLong)
    LOAD_PYTHON_SYMBOL(PyCObject_AsVoidPtr)
    LOAD_PYTHON_SYMBOL(Py_CompileString)
  }
  LOAD_PYTHON_SYMBOL(PyCapsule_New)
  LOAD_PYTHON_SYMBOL(PyCapsule_GetPointer)
  LOAD_PYTHON_SYMBOL(PyCapsule_SetContext)
  LOAD_PYTHON_SYMBOL(PyCapsule_GetContext)
  LOAD_PYTHON_SYMBOL(Py_BuildValue)

  // LOAD_PYTHON_SYMBOL(PyIter_Check) // only available beginning in 3.10
  // Try to load the symbol, and if it fails, set it to the internal function
  if (!loadSymbol(pLib_, "PyIter_Check", (void**)&PyIter_Check, NULL)) {
    PyIter_Check = &_PyIter_Check;
  }

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
    PyErr_Clear();
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
  // we aim to compile a single binary compatible with both numpy 2.x and 1.x
  PyArray_RUNTIME_VERSION = PyArray_GetNDArrayCVersion();
  if (NPY_VERSION_2 != PyArray_RUNTIME_VERSION &&
      NPY_VERSION_1 != PyArray_RUNTIME_VERSION) {
    std::ostringstream ostr;
    ostr << "incompatible NumPy binary version " << (int) PyArray_GetNDArrayCVersion() << " "
    "(expecting version " << (int) NPY_VERSION_2 << " or " << (int) NPY_VERSION_1 << ")";
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

// returns 'true' if the buffer was flushed, or if the stdout / stderr
// objects within 'sys' did not contain 'flush' methods
bool flush_std_buffer(const char* name) {

  // returns borrowed reference
  PyObject* buffer(PySys_GetObject(name));
  if (buffer == NULL || buffer == Py_None)
    return true;

  // try to invoke flush method
  if (!PyObject_HasAttrString(buffer, "flush"))
    return true;

  PyObject* result = PyObject_CallMethod(buffer, "flush", NULL);
  if (result != NULL) {
    Py_DecRef(result);
    return true;
  }

  // if we got here, an error must have occurred; print it
  PyObject *ptype, *pvalue, *ptraceback;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
  if (pvalue) {
    PyObject* pvalue_str = PyObject_Str(pvalue);
    if (pvalue_str) {
      REprintf("Error flushing Python %s: %s\n", name, PyUnicode_AsUTF8(pvalue_str));
      Py_DecRef(pvalue_str);
    }
  }

  // clean up
  if (ptype)      Py_DecRef(ptype);
  if (pvalue)     Py_DecRef(pvalue);
  if (ptraceback) Py_DecRef(ptraceback);

  return false;

}

int flush_std_buffers() {

  PyObject *error_type, *error_value, *error_traceback;
  PyErr_Fetch(&error_type, &error_value, &error_traceback);
  bool stdout_ok = flush_std_buffer("stdout");
  bool stderr_ok = flush_std_buffer("stderr");
  bool ok = stdout_ok && stderr_ok;
  PyErr_Restore(error_type, error_value, error_traceback);

  return ok ? 0 : -1;

}


} // namespace libpython
} // namespace reticulate
