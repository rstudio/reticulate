
#ifndef __LIBPYTHON_HPP__
#define __LIBPYTHON_HPP__

#include <string>

// TODO: make sure these are correct!
typedef struct _object PyObject;
typedef long Py_ssize_t;

extern void (*_Py_Initialize)();

extern PyObject* (*_PyObject_GetAttrString)(PyObject*, const char *);
extern int (*_PyObject_HasAttrString)(PyObject*, const char *);

extern Py_ssize_t (*_PyTuple_Size)(PyObject *);
extern PyObject* (*_PyTuple_GetItem)(PyObject *, Py_ssize_t);

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

