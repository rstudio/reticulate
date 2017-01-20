
#ifndef __LIBPYTHON_HPP__
#define __LIBPYTHON_HPP__

#include <string>

#define _PYTHON_API_VERSION 1013
#define _PYTHON3_ABI_VERSION 3

#if _WIN32 || _WIN64
#if _WIN64
typedef __int64 Py_ssize_t;
#else
typedef int Py_ssize_t;
#endif
#else
typedef long Py_ssize_t;
#endif

typedef struct _object PyObject;

#define METH_VARARGS  0x0001
#define METH_KEYWORDS 0x0002

#define __PyObject_HEAD_EXTRA
#define __PyObject_EXTRA_INIT

#define _PyObject_HEAD  \
__PyObject_HEAD_EXTRA   \
  Py_ssize_t ob_refcnt; \
struct __typeobject *ob_type;

#define _PyObject_VAR_HEAD               \
_PyObject_HEAD                           \
  Py_ssize_t ob_size;

typedef struct __typeobject {
_PyObject_VAR_HEAD
  const char *tp_name;
  Py_ssize_t tp_basicsize, tp_itemsize;
} _PyTypeObject;

typedef struct __object {
_PyObject_HEAD
} _PyObject;

typedef struct {
_PyObject_VAR_HEAD
} _PyVarObject;

typedef _PyObject *(*_PyCFunction)(PyObject *, PyObject *);

struct _PyMethodDef {
  const char	*ml_name;
  _PyCFunction  ml_meth;
  int		 ml_flags;
  const char	*ml_doc;
};
typedef struct _PyMethodDef _PyMethodDef;

#define _PyObject_HEAD3 _PyObject ob_base;

#define _PyObject_HEAD_INIT(type) \
{ __PyObject_EXTRA_INIT           \
  1, type },

#define _PyModuleDef_HEAD_INIT { \
_PyObject_HEAD_INIT(NULL) \
  NULL, \
  0, \
  NULL, \
}

typedef int (*_inquiry)(PyObject *);
typedef int (*_visitproc)(PyObject *, void *);
typedef int (*_traverseproc)(PyObject *, _visitproc, void *);
typedef void (*_freefunc)(void *);

typedef struct _PyModuleDef_Base {
  _PyObject_HEAD3
  _PyObject* (*m_init)(void);
  Py_ssize_t m_index;
  _PyObject* m_copy;
} _PyModuleDef_Base;

typedef struct _PyModuleDef{
  _PyModuleDef_Base m_base;
  const char* m_name;
  const char* m_doc;
  Py_ssize_t m_size;
  _PyMethodDef *m_methods;
  _inquiry m_reload;
  _traverseproc m_traverse;
  _inquiry m_clear;
  _freefunc m_free;
} _PyModuleDef;


extern PyObject* _Py_None;
extern PyObject* _Py_Unicode;
extern PyObject* _Py_String;
extern PyObject* _Py_Int;
extern PyObject* _Py_Long;
extern PyObject* _Py_Bool;
extern PyObject* _Py_True;
extern PyObject* _Py_False;
extern PyObject* _Py_Dict;
extern PyObject* _Py_Float;

#define _Py_TYPE(ob) (((PyObject*)(ob))->ob_type)

#define _PyUnicode_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_Unicode))
#define _PyString_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_String))
#define _PyInt_Check(o)  (_Py_TYPE(o) == _Py_TYPE(_Py_Int))
#define _PyLong_Check(o)  (_Py_TYPE(o) == _Py_TYPE(_Py_Long))
#define _PyBool_Check(o) ((o == _Py_False) | (o == _Py_True))
#define _PyDict_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_Dict))
#define _PyFloat_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_Float))

extern void (*_Py_Initialize)();

extern PyObject* (*_Py_InitModule4)(const char *name, _PyMethodDef *methods,
           const char *doc, PyObject *self,
           int apiver);

extern PyObject* (*_PyModule_Create2)(_PyModuleDef *def, int);
extern int (*_PyImport_AppendInittab)(const char *name, PyObject* (*initfunc)());

extern PyObject* (*_Py_BuildValue)(const char *format, ...);

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

extern int (*_PyString_AsStringAndSize)(
    register PyObject *obj,	/* string or Unicode object */
    register char **s,		/* pointer to buffer variable */
    register Py_ssize_t *len	/* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);

extern PyObject* (*_PyString_FromString)(const char *);
extern PyObject* (*_PyString_FromStringAndSize)(const char *, Py_ssize_t);

extern PyObject* (*_PyUnicode_EncodeLocale)(PyObject *unicode, const char *errors);
extern int (*_PyBytes_AsStringAndSize)(
    PyObject *obj,      /* string or Unicode object */
    char **s,           /* pointer to buffer variable */
    Py_ssize_t *len     /* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);
extern PyObject* (*_PyBytes_FromStringAndSize)(const char *, Py_ssize_t);
extern PyObject* (*_PyUnicode_FromString)(const char *u);

extern void (*_PyErr_Fetch)(PyObject **, PyObject **, PyObject **);
extern PyObject* (*_PyErr_Occurred)(void);
extern void (*_PyErr_NormalizeException)(PyObject**, PyObject**, PyObject**);

extern int (*_PyCallable_Check)(PyObject *);

extern PyObject* (*_PyModule_GetDict)(PyObject *);
extern PyObject* (*_PyImport_AddModule)(const char *);

extern PyObject* (*_PyRun_StringFlags)(const char *, int, PyObject*, PyObject*, void*);
extern int (*_PyRun_SimpleFileExFlags)(FILE *, const char *, int, void *);

extern PyObject* (*_PyObject_GetIter)(PyObject *);
extern PyObject* (*_PyIter_Next)(PyObject *);

extern void (*_PySys_SetArgv)(int, char **);
extern void (*_PySys_SetArgv_v3)(int, wchar_t **);

typedef void (*_PyCapsule_Destructor)(PyObject *);
extern PyObject* (*_PyCapsule_New)(void *pointer, const char *name, _PyCapsule_Destructor destructor);
extern void* (*_PyCapsule_GetPointer)(PyObject *capsule, const char *name);

extern PyObject* (*_PyDict_New)(void);
extern int (*_PyDict_SetItem)(PyObject *mp, PyObject *key, PyObject *item);
extern int (*_PyDict_SetItemString)(PyObject *dp, const char *key, PyObject *item);
extern int (*__PyDict_Next)(
    PyObject *mp, Py_ssize_t *pos, PyObject **key, PyObject **value);

extern PyObject* (*_PyInt_FromLong)(long);
extern long (*_PyInt_AsLong)(PyObject *);
extern PyObject* (*_PyLong_FromLong)(long);
extern long (*_PyLong_AsLong)(PyObject *);

extern PyObject* (*_PyBool_FromLong)(long);

extern PyObject* (*_PyFloat_FromDouble)(double);
extern double (*_PyFloat_AsDouble)(PyObject *);

class LibPython {

public:
  bool load(const std::string& libPath, bool python3, std::string* pError);
  bool unload(std::string* pError);

private:
  friend LibPython& libPython();
  LibPython() : pLib_(NULL) {}
  LibPython(const LibPython&);
  void* pLib_;
};

inline LibPython& libPython() {
  static LibPython instance;
  return instance;
}


#endif

