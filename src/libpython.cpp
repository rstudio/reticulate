
#include "libpython.hpp"

#ifndef _WIN32
#include <dlfcn.h>
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


#define LOAD_PYTHON_SYMBOL(name)                                \
if (!loadSymbol(pLib_, #name, (void**)&_##name, pError)) \
  return false;

bool LibPython::load(const std::string& libPath, bool python3, std::string* pError)
{
  if (!loadLibrary(libPath, &pLib_, pError))
    return false;

  LOAD_PYTHON_SYMBOL(Py_Initialize)


  return true;
}

bool LibPython::unload(std::string* pError)
{
  if (pLib_ != NULL)
    return closeLibrary(pLib_, pError);
  else
    return true;
}

void (*_Py_Initialize)();










