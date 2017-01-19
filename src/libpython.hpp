
#ifndef __LIBPYTHON_HPP__
#define __LIBPYTHON_HPP__

#include <string>

// TODO: make sure these are correct!
typedef struct _object PyObject;
typedef long Py_ssize_t;

extern void (*_Py_Initialize)();

extern void (*_Py_IncRef)(PyObject *);
extern void (*_Py_DecRef)(PyObject *);

extern PyObject* (*__PyObject_Str)(PyObject *);

extern PyObject* (*_PyObject_Dir)(PyObject *);

extern PyObject* (*_PyObject_GetAttrString)(PyObject*, const char *);
extern int (*_PyObject_HasAttrString)(PyObject*, const char *);

extern Py_ssize_t (*_PyTuple_Size)(PyObject *);
extern PyObject* (*_PyTuple_GetItem)(PyObject *, Py_ssize_t);

extern PyObject* (*_PyList_New)(Py_ssize_t size);
extern Py_ssize_t (*_PyList_Size)(PyObject *);
extern PyObject* (*_PyList_GetItem)(PyObject *, Py_ssize_t);
extern int (*_PyList_SetItem)(PyObject *, Py_ssize_t, PyObject *);

extern void (*_PyErr_Fetch)(PyObject **, PyObject **, PyObject **);
extern void (*_PyErr_NormalizeException)(PyObject**, PyObject**, PyObject**);

extern int (*_PyCallable_Check)(PyObject *);

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

