
#ifndef __LIBPYTHON_HPP__
#define __LIBPYTHON_HPP__

#include <string>

#define _PYTHON_API_VERSION 1013

#if _WIN32 || _WIN64
#if _WIN64
typedef __int64 Py_ssize_t;
#else
typedef int Py_ssize_t;
#endif
#else
typedef long Py_ssize_t;
#endif

typedef struct _object _PyObject;

#define METH_VARARGS  0x0001
#define METH_KEYWORDS 0x0002

typedef _PyObject *(*_PyCFunction)(_PyObject *, _PyObject *);
struct _PyMethodDef {
  const char	*ml_name;	/* The name of the built-in function/method */
  _PyCFunction  ml_meth;	/* The C function that implements it */
  int		 ml_flags;	/* Combination of METH_xxx flags, which mostly
 describe the args expected by the C func */
  const char	*ml_doc;	/* The __doc__ attribute, or NULL */
};
typedef struct _PyMethodDef _PyMethodDef;


extern void (*_Py_Initialize)();

extern _PyObject* (*_Py_InitModule4)(const char *name, _PyMethodDef *methods,
           const char *doc, _PyObject *self,
           int apiver);

extern void (*_Py_IncRef)(_PyObject *);
extern void (*_Py_DecRef)(_PyObject *);

extern _PyObject* (*__PyObject_Str)(_PyObject *);

extern _PyObject* (*_PyObject_Dir)(_PyObject *);

extern _PyObject* (*_PyObject_GetAttrString)(_PyObject*, const char *);
extern int (*_PyObject_HasAttrString)(_PyObject*, const char *);

extern Py_ssize_t (*_PyTuple_Size)(_PyObject *);
extern _PyObject* (*_PyTuple_GetItem)(_PyObject *, Py_ssize_t);

extern _PyObject* (*_PyList_New)(Py_ssize_t size);
extern Py_ssize_t (*_PyList_Size)(_PyObject *);
extern _PyObject* (*_PyList_GetItem)(_PyObject *, Py_ssize_t);
extern int (*_PyList_SetItem)(_PyObject *, Py_ssize_t, _PyObject *);

extern int (*_PyString_AsStringAndSize)(
    register _PyObject *obj,	/* string or Unicode object */
    register char **s,		/* pointer to buffer variable */
    register Py_ssize_t *len	/* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);

extern _PyObject* (*_PyString_FromString)(const char *);
extern _PyObject* (*_PyString_FromStringAndSize)(const char *, Py_ssize_t);


extern void (*_PyErr_Fetch)(_PyObject **, _PyObject **, _PyObject **);
extern _PyObject* (*_PyErr_Occurred)(void);
extern void (*_PyErr_NormalizeException)(_PyObject**, _PyObject**, _PyObject**);

extern int (*_PyCallable_Check)(_PyObject *);

extern _PyObject* (*_PyModule_GetDict)(_PyObject *);
extern _PyObject* (*_PyImport_AddModule)(const char *);

extern _PyObject* (*_PyRun_StringFlags)(const char *, int, _PyObject*, _PyObject*, void*);
extern int (*_PyRun_SimpleFileExFlags)(FILE *, const char *, int, void *);

extern _PyObject* (*_PyObject_GetIter)(_PyObject *);
extern _PyObject* (*_PyIter_Next)(_PyObject *);

extern void (*_PySys_SetArgv)(int, char **);

typedef void (*_PyCapsule_Destructor)(_PyObject *);
extern _PyObject* (*_PyCapsule_New)(void *pointer, const char *name, _PyCapsule_Destructor destructor);
extern void* (*_PyCapsule_GetPointer)(_PyObject *capsule, const char *name);

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

