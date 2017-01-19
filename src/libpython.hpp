
#ifndef __LIBPYTHON_HPP__
#define __LIBPYTHON_HPP__

#include <string>

typedef struct _object PyObject;

extern void (*_Py_Initialize)();
extern PyObject* (*_PyObject_GetAttrString)(PyObject*, const char *);


class LibPython {

public:
  LibPython() : pLib_(NULL) {}
  bool load(const std::string& libPath, bool python3, std::string* pError);
  bool unload(std::string* pError);

private:
  LibPython(const LibPython&);
  void* pLib_;
};

#endif

