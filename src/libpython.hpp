
#ifndef __LIBPYTHON_HPP__
#define __LIBPYTHON_HPP__

#include <string>

#ifndef LIBPYTHON_CPP
#define LIBPYTHON_EXTERN extern
#else
#define LIBPYTHON_EXTERN
#endif

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


LIBPYTHON_EXTERN _PyTypeObject* _PyFunction_Type;
LIBPYTHON_EXTERN _PyTypeObject* _PyModule_Type;

LIBPYTHON_EXTERN PyObject* _Py_None;
LIBPYTHON_EXTERN PyObject* _Py_Unicode;
LIBPYTHON_EXTERN PyObject* _Py_String;
LIBPYTHON_EXTERN PyObject* _Py_Int;
LIBPYTHON_EXTERN PyObject* _Py_Long;
LIBPYTHON_EXTERN PyObject* _Py_Bool;
LIBPYTHON_EXTERN PyObject* _Py_True;
LIBPYTHON_EXTERN PyObject* _Py_False;
LIBPYTHON_EXTERN PyObject* _Py_Dict;
LIBPYTHON_EXTERN PyObject* _Py_Float;
LIBPYTHON_EXTERN PyObject* _Py_List;
LIBPYTHON_EXTERN PyObject* _Py_Tuple;
LIBPYTHON_EXTERN PyObject* _Py_Complex;

#define _Py_TYPE(ob) (((PyObject*)(ob))->ob_type)

#define _PyUnicode_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_Unicode))
#define _PyString_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_String))
#define _PyInt_Check(o)  (_Py_TYPE(o) == _Py_TYPE(_Py_Int))
#define _PyLong_Check(o)  (_Py_TYPE(o) == _Py_TYPE(_Py_Long))
#define _PyBool_Check(o) ((o == _Py_False) | (o == _Py_True))
#define _PyDict_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_Dict))
#define _PyFloat_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_Float))
#define _PyFunction_Check(op) ((_PyTypeObject*)(_Py_TYPE(op)) == _PyFunction_Type)
#define _PyTuple_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_Tuple))
#define _PyList_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_List))
#define _PyComplex_Check(o) (_Py_TYPE(o) == _Py_TYPE(_Py_Complex))

LIBPYTHON_EXTERN void (*_Py_Initialize)();

LIBPYTHON_EXTERN PyObject* (*_Py_InitModule4)(const char *name, _PyMethodDef *methods,
           const char *doc, PyObject *self,
           int apiver);

LIBPYTHON_EXTERN PyObject* (*_PyImport_ImportModule)(const char *name);

LIBPYTHON_EXTERN PyObject* (*_PyModule_Create2)(_PyModuleDef *def, int);
LIBPYTHON_EXTERN int (*_PyImport_AppendInittab)(const char *name, PyObject* (*initfunc)());

LIBPYTHON_EXTERN PyObject* (*_Py_BuildValue)(const char *format, ...);

LIBPYTHON_EXTERN void (*_Py_IncRef)(PyObject *);
LIBPYTHON_EXTERN void (*_Py_DecRef)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*__PyObject_Str)(PyObject *);

LIBPYTHON_EXTERN int (*_PyObject_IsInstance)(PyObject *object, PyObject *typeorclass);

LIBPYTHON_EXTERN PyObject* (*_PyObject_Dir)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*_PyObject_Call)(PyObject *callable_object,
           PyObject *args, PyObject *kw);
LIBPYTHON_EXTERN PyObject* (*_PyObject_CallFunctionObjArgs)(PyObject *callable,
           ...);

LIBPYTHON_EXTERN PyObject* (*_PyObject_GetAttrString)(PyObject*, const char *);
LIBPYTHON_EXTERN int (*_PyObject_HasAttrString)(PyObject*, const char *);

LIBPYTHON_EXTERN Py_ssize_t (*_PyTuple_Size)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*_PyTuple_GetItem)(PyObject *, Py_ssize_t);
LIBPYTHON_EXTERN PyObject* (*_PyTuple_New)(Py_ssize_t size);
LIBPYTHON_EXTERN int (*_PyTuple_SetItem)(PyObject *, Py_ssize_t, PyObject *);
LIBPYTHON_EXTERN PyObject* (*_PyTuple_GetSlice)(PyObject *, Py_ssize_t, Py_ssize_t);

LIBPYTHON_EXTERN PyObject* (*_PyList_New)(Py_ssize_t size);
LIBPYTHON_EXTERN Py_ssize_t (*_PyList_Size)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*_PyList_GetItem)(PyObject *, Py_ssize_t);
LIBPYTHON_EXTERN int (*_PyList_SetItem)(PyObject *, Py_ssize_t, PyObject *);

LIBPYTHON_EXTERN int (*_PyString_AsStringAndSize)(
    register PyObject *obj,	/* string or Unicode object */
    register char **s,		/* pointer to buffer variable */
    register Py_ssize_t *len	/* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);

LIBPYTHON_EXTERN PyObject* (*_PyString_FromString)(const char *);
LIBPYTHON_EXTERN PyObject* (*_PyString_FromStringAndSize)(const char *, Py_ssize_t);

LIBPYTHON_EXTERN PyObject* (*_PyUnicode_EncodeLocale)(PyObject *unicode, const char *errors);
LIBPYTHON_EXTERN int (*_PyBytes_AsStringAndSize)(
    PyObject *obj,      /* string or Unicode object */
    char **s,           /* pointer to buffer variable */
    Py_ssize_t *len     /* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);
LIBPYTHON_EXTERN PyObject* (*_PyBytes_FromStringAndSize)(const char *, Py_ssize_t);
LIBPYTHON_EXTERN PyObject* (*_PyUnicode_FromString)(const char *u);

LIBPYTHON_EXTERN void (*_PyErr_Fetch)(PyObject **, PyObject **, PyObject **);
LIBPYTHON_EXTERN PyObject* (*_PyErr_Occurred)(void);
LIBPYTHON_EXTERN void (*_PyErr_NormalizeException)(PyObject**, PyObject**, PyObject**);

LIBPYTHON_EXTERN int (*_PyCallable_Check)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*_PyModule_GetDict)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*_PyImport_AddModule)(const char *);

LIBPYTHON_EXTERN PyObject* (*_PyRun_StringFlags)(const char *, int, PyObject*, PyObject*, void*);
LIBPYTHON_EXTERN int (*_PyRun_SimpleFileExFlags)(FILE *, const char *, int, void *);

LIBPYTHON_EXTERN PyObject* (*_PyObject_GetIter)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*_PyIter_Next)(PyObject *);

LIBPYTHON_EXTERN void (*_PySys_SetArgv)(int, char **);
LIBPYTHON_EXTERN void (*_PySys_SetArgv_v3)(int, wchar_t **);

typedef void (*_PyCapsule_Destructor)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*_PyCapsule_New)(void *pointer, const char *name, _PyCapsule_Destructor destructor);
LIBPYTHON_EXTERN void* (*_PyCapsule_GetPointer)(PyObject *capsule, const char *name);

LIBPYTHON_EXTERN PyObject* (*_PyDict_New)(void);
LIBPYTHON_EXTERN int (*_PyDict_SetItem)(PyObject *mp, PyObject *key, PyObject *item);
LIBPYTHON_EXTERN int (*_PyDict_SetItemString)(PyObject *dp, const char *key, PyObject *item);
LIBPYTHON_EXTERN int (*__PyDict_Next)(
    PyObject *mp, Py_ssize_t *pos, PyObject **key, PyObject **value);

LIBPYTHON_EXTERN PyObject* (*_PyInt_FromLong)(long);
LIBPYTHON_EXTERN long (*_PyInt_AsLong)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*_PyLong_FromLong)(long);
LIBPYTHON_EXTERN long (*_PyLong_AsLong)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*_PyBool_FromLong)(long);

LIBPYTHON_EXTERN PyObject* (*_PyFloat_FromDouble)(double);
LIBPYTHON_EXTERN double (*_PyFloat_AsDouble)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*_PyComplex_FromDoubles)(double real, double imag);
LIBPYTHON_EXTERN double (*_PyComplex_RealAsDouble)(PyObject *op);
LIBPYTHON_EXTERN double (*_PyComplex_ImagAsDouble)(PyObject *op);


enum _NPY_TYPES {
  _NPY_BOOL=0,
  _NPY_BYTE, _NPY_UBYTE,
  _NPY_SHORT, _NPY_USHORT,
  _NPY_INT, _NPY_UINT,
  _NPY_LONG, _NPY_ULONG,
  _NPY_LONGLONG, _NPY_ULONGLONG,
  _NPY_FLOAT, _NPY_DOUBLE, _NPY_LONGDOUBLE,
  _NPY_CFLOAT, _NPY_CDOUBLE, _NPY_CLONGDOUBLE,
  _NPY_OBJECT=17,
  _NPY_STRING, _NPY_UNICODE,
  _NPY_VOID,
  _NPY_DATETIME, _NPY_TIMEDELTA, _NPY_HALF,
  _NPY_NTYPES,
  _NPY_NOTYPE,
  _NPY_CHAR,
  _NPY_USERDEF=256,
  _NPY_NTYPES_ABI_COMPATIBLE=21
};

// int is still 32 bits on all relevant 64-bit platforms
#define _NPY_INT32 _NPY_INT


class SharedLibrary {

public:
  bool load(const std::string& libPath, bool python3, std::string* pError);
  bool unload(std::string* pError);
  virtual ~SharedLibrary() {}

private:
  virtual bool loadSymbols(bool python3, std::string* pError) = 0;

protected:
  SharedLibrary() : pLib_(NULL) {}
private:
  SharedLibrary(const SharedLibrary&);

protected:
  void* pLib_;
};

class LibPython : public SharedLibrary {
private:
  LibPython() : SharedLibrary() {}
  friend SharedLibrary& libPython();
  virtual bool loadSymbols(bool python3, std::string* pError);
};

inline SharedLibrary& libPython() {
  static LibPython instance;
  return instance;
}

class LibNumPy : public SharedLibrary {
private:
  LibNumPy() : SharedLibrary() {}
  friend SharedLibrary& libNumPy();
  virtual bool loadSymbols(bool python3, std::string* pError);
};

inline SharedLibrary& libNumPy() {
  static LibNumPy instance;
  return instance;
}

#endif

