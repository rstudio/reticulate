
#ifndef __LIBPYTHON_HPP__
#define __LIBPYTHON_HPP__

#include <string>
#include <stdint.h>

#ifndef LIBPYTHON_CPP
#define LIBPYTHON_EXTERN extern
#else
#define LIBPYTHON_EXTERN
#endif

#define _PYTHON_API_VERSION 1013
#define _PYTHON3_ABI_VERSION 3

namespace libpython {

#if _WIN32 || _WIN64
#if _WIN64
typedef __int64 Py_ssize_t;
#else
typedef int Py_ssize_t;
#endif
#else
typedef long Py_ssize_t;
#endif

#define METH_VARARGS  0x0001
#define METH_KEYWORDS 0x0002

#define Py_file_input 257

#define _PyObject_HEAD_EXTRA
#define _PyObject_EXTRA_INIT

#define PyObject_HEAD  \
_PyObject_HEAD_EXTRA   \
  Py_ssize_t ob_refcnt; \
struct _typeobject *ob_type;

#define PyObject_VAR_HEAD               \
PyObject_HEAD                           \
  Py_ssize_t ob_size;

typedef struct _typeobject {
PyObject_VAR_HEAD
  const char *tp_name;
  Py_ssize_t tp_basicsize, tp_itemsize;
} PyTypeObject;

typedef struct _object {
PyObject_HEAD
} PyObject;

typedef PyObject *(*PyCFunction)(PyObject *, PyObject *);

struct PyMethodDef {
  const char	*ml_name;
  PyCFunction  ml_meth;
  int		 ml_flags;
  const char	*ml_doc;
};
typedef struct PyMethodDef PyMethodDef;

#define PyObject_HEAD3 PyObject ob_base;

#define PyObject_HEAD_INIT(type) \
{ _PyObject_EXTRA_INIT           \
  1, type },

#define PyModuleDef_HEAD_INIT { \
PyObject_HEAD_INIT(NULL) \
  NULL, \
  0, \
  NULL, \
}

typedef int (*inquiry)(PyObject *);
typedef int (*visitproc)(PyObject *, void *);
typedef int (*traverseproc)(PyObject *, visitproc, void *);
typedef void (*freefunc)(void *);

typedef struct PyModuleDef_Base {
  PyObject_HEAD3
  PyObject* (*m_init)(void);
  Py_ssize_t m_index;
  PyObject* m_copy;
} PyModuleDef_Base;

typedef struct PyModuleDef{
  PyModuleDef_Base m_base;
  const char* m_name;
  const char* m_doc;
  Py_ssize_t m_size;
  PyMethodDef *m_methods;
  inquiry m_reload;
  traverseproc m_traverse;
  inquiry m_clear;
  freefunc m_free;
} PyModuleDef;


LIBPYTHON_EXTERN PyTypeObject* PyFunction_Type;
LIBPYTHON_EXTERN PyTypeObject* PyModule_Type;

LIBPYTHON_EXTERN PyObject* Py_None;
LIBPYTHON_EXTERN PyObject* Py_Unicode;
LIBPYTHON_EXTERN PyObject* Py_String;
LIBPYTHON_EXTERN PyObject* Py_Int;
LIBPYTHON_EXTERN PyObject* Py_Long;
LIBPYTHON_EXTERN PyObject* Py_Bool;
LIBPYTHON_EXTERN PyObject* Py_True;
LIBPYTHON_EXTERN PyObject* Py_False;
LIBPYTHON_EXTERN PyObject* Py_Dict;
LIBPYTHON_EXTERN PyObject* Py_Float;
LIBPYTHON_EXTERN PyObject* Py_List;
LIBPYTHON_EXTERN PyObject* Py_Tuple;
LIBPYTHON_EXTERN PyObject* Py_Complex;

#define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)

#define PyUnicode_Check(o) (Py_TYPE(o) == Py_TYPE(Py_Unicode))
#define PyString_Check(o) (Py_TYPE(o) == Py_TYPE(Py_String))
#define PyInt_Check(o)  (Py_TYPE(o) == Py_TYPE(Py_Int))
#define PyLong_Check(o)  (Py_TYPE(o) == Py_TYPE(Py_Long))
#define PyBool_Check(o) ((o == Py_False) | (o == Py_True))
#define PyDict_Check(o) (Py_TYPE(o) == Py_TYPE(Py_Dict))
#define PyFloat_Check(o) (Py_TYPE(o) == Py_TYPE(Py_Float))
#define PyFunction_Check(op) ((PyTypeObject*)(Py_TYPE(op)) == PyFunction_Type)
#define PyTuple_Check(o) (Py_TYPE(o) == Py_TYPE(Py_Tuple))
#define PyList_Check(o) (Py_TYPE(o) == Py_TYPE(Py_List))
#define PyComplex_Check(o) (Py_TYPE(o) == Py_TYPE(Py_Complex))

LIBPYTHON_EXTERN void (*Py_Initialize)();

LIBPYTHON_EXTERN PyObject* (*Py_InitModule4)(const char *name, PyMethodDef *methods,
           const char *doc, PyObject *self,
           int apiver);

LIBPYTHON_EXTERN PyObject* (*PyImport_ImportModule)(const char *name);

LIBPYTHON_EXTERN PyObject* (*PyModule_Create2)(PyModuleDef *def, int);
LIBPYTHON_EXTERN int (*PyImport_AppendInittab)(const char *name, PyObject* (*initfunc)());

LIBPYTHON_EXTERN PyObject* (*Py_BuildValue)(const char *format, ...);

LIBPYTHON_EXTERN void (*Py_IncRef)(PyObject *);
LIBPYTHON_EXTERN void (*Py_DecRef)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*PyObject_Str)(PyObject *);

LIBPYTHON_EXTERN int (*PyObject_IsInstance)(PyObject *object, PyObject *typeorclass);

LIBPYTHON_EXTERN PyObject* (*PyObject_Dir)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*PyObject_Call)(PyObject *callable_object,
           PyObject *args, PyObject *kw);
LIBPYTHON_EXTERN PyObject* (*PyObject_CallFunctionObjArgs)(PyObject *callable,
           ...);

LIBPYTHON_EXTERN PyObject* (*PyObject_GetAttrString)(PyObject*, const char *);
LIBPYTHON_EXTERN int (*PyObject_HasAttrString)(PyObject*, const char *);

LIBPYTHON_EXTERN Py_ssize_t (*PyTuple_Size)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyTuple_GetItem)(PyObject *, Py_ssize_t);
LIBPYTHON_EXTERN PyObject* (*PyTuple_New)(Py_ssize_t size);
LIBPYTHON_EXTERN int (*PyTuple_SetItem)(PyObject *, Py_ssize_t, PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyTuple_GetSlice)(PyObject *, Py_ssize_t, Py_ssize_t);

LIBPYTHON_EXTERN PyObject* (*PyList_New)(Py_ssize_t size);
LIBPYTHON_EXTERN Py_ssize_t (*PyList_Size)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyList_GetItem)(PyObject *, Py_ssize_t);
LIBPYTHON_EXTERN int (*PyList_SetItem)(PyObject *, Py_ssize_t, PyObject *);

LIBPYTHON_EXTERN int (*PyString_AsStringAndSize)(
    register PyObject *obj,	/* string or Unicode object */
    register char **s,		/* pointer to buffer variable */
    register Py_ssize_t *len	/* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);

LIBPYTHON_EXTERN PyObject* (*PyString_FromString)(const char *);
LIBPYTHON_EXTERN PyObject* (*PyString_FromStringAndSize)(const char *, Py_ssize_t);

LIBPYTHON_EXTERN PyObject* (*PyUnicode_EncodeLocale)(PyObject *unicode, const char *errors);
LIBPYTHON_EXTERN int (*PyBytes_AsStringAndSize)(
    PyObject *obj,      /* string or Unicode object */
    char **s,           /* pointer to buffer variable */
    Py_ssize_t *len     /* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);
LIBPYTHON_EXTERN PyObject* (*PyBytes_FromStringAndSize)(const char *, Py_ssize_t);
LIBPYTHON_EXTERN PyObject* (*PyUnicode_FromString)(const char *u);

LIBPYTHON_EXTERN void (*PyErr_Fetch)(PyObject **, PyObject **, PyObject **);
LIBPYTHON_EXTERN PyObject* (*PyErr_Occurred)(void);
LIBPYTHON_EXTERN void (*PyErr_NormalizeException)(PyObject**, PyObject**, PyObject**);

LIBPYTHON_EXTERN int (*PyCallable_Check)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*PyModule_GetDict)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyImport_AddModule)(const char *);

LIBPYTHON_EXTERN PyObject* (*PyRun_StringFlags)(const char *, int, PyObject*, PyObject*, void*);
LIBPYTHON_EXTERN int (*PyRun_SimpleFileExFlags)(FILE *, const char *, int, void *);

LIBPYTHON_EXTERN PyObject* (*PyObject_GetIter)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyIter_Next)(PyObject *);

LIBPYTHON_EXTERN void (*PySys_SetArgv)(int, char **);
LIBPYTHON_EXTERN void (*PySys_SetArgv_v3)(int, wchar_t **);

typedef void (*PyCapsule_Destructor)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyCapsule_New)(void *pointer, const char *name, PyCapsule_Destructor destructor);
LIBPYTHON_EXTERN void* (*PyCapsule_GetPointer)(PyObject *capsule, const char *name);

LIBPYTHON_EXTERN PyObject* (*PyDict_New)(void);
LIBPYTHON_EXTERN int (*PyDict_SetItem)(PyObject *mp, PyObject *key, PyObject *item);
LIBPYTHON_EXTERN int (*PyDict_SetItemString)(PyObject *dp, const char *key, PyObject *item);
LIBPYTHON_EXTERN int (*PyDict_Next)(
    PyObject *mp, Py_ssize_t *pos, PyObject **key, PyObject **value);

LIBPYTHON_EXTERN PyObject* (*PyInt_FromLong)(long);
LIBPYTHON_EXTERN long (*PyInt_AsLong)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyLong_FromLong)(long);
LIBPYTHON_EXTERN long (*PyLong_AsLong)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*PyBool_FromLong)(long);

LIBPYTHON_EXTERN PyObject* (*PyFloat_FromDouble)(double);
LIBPYTHON_EXTERN double (*PyFloat_AsDouble)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*PyComplex_FromDoubles)(double real, double imag);
LIBPYTHON_EXTERN double (*PyComplex_RealAsDouble)(PyObject *op);
LIBPYTHON_EXTERN double (*PyComplex_ImagAsDouble)(PyObject *op);

LIBPYTHON_EXTERN void* (*PyCObject_AsVoidPtr)(PyObject *);

LIBPYTHON_EXTERN int (*PyType_IsSubtype)(PyTypeObject *, PyTypeObject *);


#define PyObject_TypeCheck(o, tp) ((PyTypeObject*)Py_TYPE(o) == (tp)) || PyType_IsSubtype((PyTypeObject*)Py_TYPE(o), (tp))

enum NPY_TYPES {
  NPY_BOOL=0,
  NPY_BYTE, NPY_UBYTE,
  NPY_SHORT, NPY_USHORT,
  NPY_INT, NPY_UINT,
  NPY_LONG, NPY_ULONG,
  NPY_LONGLONG, NPY_ULONGLONG,
  NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
  NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
  NPY_OBJECT=17,
  NPY_STRING, NPY_UNICODE,
  NPY_VOID,
  NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,
  NPY_NTYPES,
  NPY_NOTYPE,
  NPY_CHAR,
  NPY_USERDEF=256,
  NPY_NTYPES_ABI_COMPATIBLE=21
};


// PyArray_Descr is opaque to our code so we just get the header

typedef struct {
  PyObject_HEAD
  PyTypeObject *typeobj;
  char kind;
  char type;
  char byteorder;
  char flags;
  int type_num;
  int elsize;
  int alignment;

  // ...more fields here we don't capture...

} PyArray_Descr;

typedef struct tagPyArrayObject {
  PyObject_HEAD
} PyArrayObject;


typedef unsigned char npy_bool;
typedef long npy_long;
typedef double npy_double;
typedef struct { double real, imag; } npy_cdouble;
typedef npy_cdouble npy_complex128;

typedef intptr_t npy_intp;


typedef struct tagPyArrayObject_fields {
  PyObject_HEAD
  /* Pointer to the raw data buffer */
  char *data;
  /* The number of dimensions, also called 'ndim' */
  int nd;
  /* The size in each dimension, also called 'shape' */
  npy_intp *dimensions;
  /*
  * Number of bytes to jump to get to the
  * next element in each dimension
  */
  npy_intp *strides;
  /*
  * This object is decref'd upon
  * deletion of array. Except in the
  * case of UPDATEIFCOPY which has
  * special handling.
  *
  * For views it points to the original
  * array, collapsed so no chains of
  * views occur.
  *
  * For creation from buffer object it
  * points to an object that should be
  * decref'd on deletion
  *
  * For UPDATEIFCOPY flag this is an
  * array to-be-updated upon deletion
  * of this one
  */
  PyObject *base;
  /* Pointer to type structure */
  PyArray_Descr *descr;
  /* Flags describing array -- see below */
  int flags;
  /* For weak references */
  PyObject *weakreflist;
} PyArrayObject_fields;



LIBPYTHON_EXTERN void **PyArray_API;

#define PyArray_Type (*(PyTypeObject *)PyArray_API[2])

#define PyGenericArrType_Type (*(PyTypeObject *)PyArray_API[10])

#define PyArray_CastToType                                \
(*(PyObject * (*)(PyArrayObject *, PyArray_Descr *, int)) \
   PyArray_API[49])

#define PyArray_SetBaseObject             \
 (*(int (*)(PyArrayObject *, PyObject *)) \
    PyArray_API[282])

#define PyArray_MultiplyList        \
  (*(npy_intp (*)(npy_intp *, int)) \
     PyArray_API[158])                                        \

#define PyArray_DescrFromType     \
     (*(PyArray_Descr * (*)(int)) \
        PyArray_API[45])

#define PyArray_DescrFromScalar           \
      (*(PyArray_Descr * (*)(PyObject *)) \
         PyArray_API[57])                                     \

#define PyArray_CastScalarToCtype                         \
         (*(int (*)(PyObject *, void *, PyArray_Descr *)) \
            PyArray_API[63])

#define PyArray_New                                                                                          \
          (*(PyObject * (*)(PyTypeObject *, int, npy_intp *, int, npy_intp *, void *, int, int, PyObject *)) \
             PyArray_API[93])

inline void* PyArray_DATA(PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->data;
}

inline npy_intp* PyArray_DIMS(PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->dimensions;
}

inline int PyArray_TYPE(const PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->descr->type_num;
}

inline int PyArray_NDIM(const PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->nd;
}

#define PyArray_SIZE(m) PyArray_MultiplyList(PyArray_DIMS(m), PyArray_NDIM(m))

#define PyArray_Check(o) PyObject_TypeCheck(o, &PyArray_Type)

#define PyArray_IsZeroDim(op) ((PyArray_Check(op)) && \
             (PyArray_NDIM((PyArrayObject *)op) == 0))

#define PyArray_IsScalar(obj, cls)                                            \
           (PyObject_TypeCheck(obj, &Py##cls##ArrType_Type))

#define PyArray_CheckScalar(m) (PyArray_IsScalar(m, Generic) ||               \
         (PyArray_IsZeroDim(m)))                                 \


inline bool import_numpy_api(bool python3, std::string* pError) {

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

  return true;
}

#define NPY_ARRAY_F_CONTIGUOUS    0x0002
#define NPY_ARRAY_ALIGNED         0x0100
#define NPY_ARRAY_FARRAY_RO    (NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED)

#define NPY_ARRAY_WRITEABLE       0x0400
#define NPY_ARRAY_BEHAVED      (NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE)
#define NPY_ARRAY_FARRAY       (NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED)

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

} // namespace libpython

#endif

