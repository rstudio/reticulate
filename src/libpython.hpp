
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

#define METH_VARARGS  0x0001
#define METH_KEYWORDS 0x0002

#define _Py_file_input 257

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

typedef _PyObject *(*_PyCFunction)(_PyObject *, _PyObject *);

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

typedef int (*_inquiry)(_PyObject *);
typedef int (*_visitproc)(_PyObject *, void *);
typedef int (*_traverseproc)(_PyObject *, _visitproc, void *);
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

LIBPYTHON_EXTERN _PyObject* _Py_None;
LIBPYTHON_EXTERN _PyObject* _Py_Unicode;
LIBPYTHON_EXTERN _PyObject* _Py_String;
LIBPYTHON_EXTERN _PyObject* _Py_Int;
LIBPYTHON_EXTERN _PyObject* _Py_Long;
LIBPYTHON_EXTERN _PyObject* _Py_Bool;
LIBPYTHON_EXTERN _PyObject* _Py_True;
LIBPYTHON_EXTERN _PyObject* _Py_False;
LIBPYTHON_EXTERN _PyObject* _Py_Dict;
LIBPYTHON_EXTERN _PyObject* _Py_Float;
LIBPYTHON_EXTERN _PyObject* _Py_List;
LIBPYTHON_EXTERN _PyObject* _Py_Tuple;
LIBPYTHON_EXTERN _PyObject* _Py_Complex;

#define _Py_TYPE(ob) (((_PyObject*)(ob))->ob_type)

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

LIBPYTHON_EXTERN _PyObject* (*_Py_InitModule4)(const char *name, _PyMethodDef *methods,
           const char *doc, _PyObject *self,
           int apiver);

LIBPYTHON_EXTERN _PyObject* (*_PyImport_ImportModule)(const char *name);

LIBPYTHON_EXTERN _PyObject* (*_PyModule_Create2)(_PyModuleDef *def, int);
LIBPYTHON_EXTERN int (*_PyImport_AppendInittab)(const char *name, _PyObject* (*initfunc)());

LIBPYTHON_EXTERN _PyObject* (*_Py_BuildValue)(const char *format, ...);

LIBPYTHON_EXTERN void (*_Py_IncRef)(_PyObject *);
LIBPYTHON_EXTERN void (*_Py_DecRef)(_PyObject *);

LIBPYTHON_EXTERN _PyObject* (*__PyObject_Str)(_PyObject *);

LIBPYTHON_EXTERN int (*_PyObject_IsInstance)(_PyObject *object, _PyObject *typeorclass);

LIBPYTHON_EXTERN _PyObject* (*_PyObject_Dir)(_PyObject *);

LIBPYTHON_EXTERN _PyObject* (*_PyObject_Call)(_PyObject *callable_object,
           _PyObject *args, _PyObject *kw);
LIBPYTHON_EXTERN _PyObject* (*_PyObject_CallFunctionObjArgs)(_PyObject *callable,
           ...);

LIBPYTHON_EXTERN _PyObject* (*_PyObject_GetAttrString)(_PyObject*, const char *);
LIBPYTHON_EXTERN int (*_PyObject_HasAttrString)(_PyObject*, const char *);

LIBPYTHON_EXTERN Py_ssize_t (*_PyTuple_Size)(_PyObject *);
LIBPYTHON_EXTERN _PyObject* (*_PyTuple_GetItem)(_PyObject *, Py_ssize_t);
LIBPYTHON_EXTERN _PyObject* (*_PyTuple_New)(Py_ssize_t size);
LIBPYTHON_EXTERN int (*_PyTuple_SetItem)(_PyObject *, Py_ssize_t, _PyObject *);
LIBPYTHON_EXTERN _PyObject* (*_PyTuple_GetSlice)(_PyObject *, Py_ssize_t, Py_ssize_t);

LIBPYTHON_EXTERN _PyObject* (*_PyList_New)(Py_ssize_t size);
LIBPYTHON_EXTERN Py_ssize_t (*_PyList_Size)(_PyObject *);
LIBPYTHON_EXTERN _PyObject* (*_PyList_GetItem)(_PyObject *, Py_ssize_t);
LIBPYTHON_EXTERN int (*_PyList_SetItem)(_PyObject *, Py_ssize_t, _PyObject *);

LIBPYTHON_EXTERN int (*_PyString_AsStringAndSize)(
    register _PyObject *obj,	/* string or Unicode object */
    register char **s,		/* pointer to buffer variable */
    register Py_ssize_t *len	/* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);

LIBPYTHON_EXTERN _PyObject* (*_PyString_FromString)(const char *);
LIBPYTHON_EXTERN _PyObject* (*_PyString_FromStringAndSize)(const char *, Py_ssize_t);

LIBPYTHON_EXTERN _PyObject* (*_PyUnicode_EncodeLocale)(_PyObject *unicode, const char *errors);
LIBPYTHON_EXTERN int (*_PyBytes_AsStringAndSize)(
    _PyObject *obj,      /* string or Unicode object */
    char **s,           /* pointer to buffer variable */
    Py_ssize_t *len     /* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);
LIBPYTHON_EXTERN _PyObject* (*_PyBytes_FromStringAndSize)(const char *, Py_ssize_t);
LIBPYTHON_EXTERN _PyObject* (*_PyUnicode_FromString)(const char *u);

LIBPYTHON_EXTERN void (*_PyErr_Fetch)(_PyObject **, _PyObject **, _PyObject **);
LIBPYTHON_EXTERN _PyObject* (*_PyErr_Occurred)(void);
LIBPYTHON_EXTERN void (*_PyErr_NormalizeException)(_PyObject**, _PyObject**, _PyObject**);

LIBPYTHON_EXTERN int (*_PyCallable_Check)(_PyObject *);

LIBPYTHON_EXTERN _PyObject* (*_PyModule_GetDict)(_PyObject *);
LIBPYTHON_EXTERN _PyObject* (*_PyImport_AddModule)(const char *);

LIBPYTHON_EXTERN _PyObject* (*_PyRun_StringFlags)(const char *, int, _PyObject*, _PyObject*, void*);
LIBPYTHON_EXTERN int (*_PyRun_SimpleFileExFlags)(FILE *, const char *, int, void *);

LIBPYTHON_EXTERN _PyObject* (*_PyObject_GetIter)(_PyObject *);
LIBPYTHON_EXTERN _PyObject* (*_PyIter_Next)(_PyObject *);

LIBPYTHON_EXTERN void (*_PySys_SetArgv)(int, char **);
LIBPYTHON_EXTERN void (*_PySys_SetArgv_v3)(int, wchar_t **);

typedef void (*_PyCapsule_Destructor)(_PyObject *);
LIBPYTHON_EXTERN _PyObject* (*_PyCapsule_New)(void *pointer, const char *name, _PyCapsule_Destructor destructor);
LIBPYTHON_EXTERN void* (*_PyCapsule_GetPointer)(_PyObject *capsule, const char *name);

LIBPYTHON_EXTERN _PyObject* (*_PyDict_New)(void);
LIBPYTHON_EXTERN int (*_PyDict_SetItem)(_PyObject *mp, _PyObject *key, _PyObject *item);
LIBPYTHON_EXTERN int (*_PyDict_SetItemString)(_PyObject *dp, const char *key, _PyObject *item);
LIBPYTHON_EXTERN int (*__PyDict_Next)(
    _PyObject *mp, Py_ssize_t *pos, _PyObject **key, _PyObject **value);

LIBPYTHON_EXTERN _PyObject* (*_PyInt_FromLong)(long);
LIBPYTHON_EXTERN long (*_PyInt_AsLong)(_PyObject *);
LIBPYTHON_EXTERN _PyObject* (*_PyLong_FromLong)(long);
LIBPYTHON_EXTERN long (*_PyLong_AsLong)(_PyObject *);

LIBPYTHON_EXTERN _PyObject* (*_PyBool_FromLong)(long);

LIBPYTHON_EXTERN _PyObject* (*_PyFloat_FromDouble)(double);
LIBPYTHON_EXTERN double (*_PyFloat_AsDouble)(_PyObject *);

LIBPYTHON_EXTERN _PyObject* (*_PyComplex_FromDoubles)(double real, double imag);
LIBPYTHON_EXTERN double (*_PyComplex_RealAsDouble)(_PyObject *op);
LIBPYTHON_EXTERN double (*_PyComplex_ImagAsDouble)(_PyObject *op);

LIBPYTHON_EXTERN void* (*_PyCObject_AsVoidPtr)(_PyObject *);

LIBPYTHON_EXTERN int (*_PyType_IsSubtype)(_PyTypeObject *, _PyTypeObject *);


#define _PyObject_TypeCheck(o, tp) ((_PyTypeObject*)_Py_TYPE(o) == (tp)) || _PyType_IsSubtype((_PyTypeObject*)_Py_TYPE(o), (tp))

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


// PyArray_Descr is opaque to our code so we just get the header

typedef struct ___PyArray_Descr {
  _PyObject_HEAD
  _PyTypeObject *typeobj;
  char kind;
  char type;
  char byteorder;
  char flags;
  int type_num;
  int elsize;
  int alignment;

  // ...more fields here we don't capture...

} __PyArray_Descr;

typedef struct _tagPyArrayObject {
  _PyObject_HEAD
} _PyArrayObject;


typedef unsigned char _npy_bool;
typedef long _npy_long;
typedef double _npy_double;
typedef struct { double real, imag; } _npy_cdouble;
typedef _npy_cdouble _npy_complex128;

typedef intptr_t _npy_intp;


typedef struct _tagPyArrayObject_fields {
  _PyObject_HEAD
  /* Pointer to the raw data buffer */
  char *data;
  /* The number of dimensions, also called 'ndim' */
  int nd;
  /* The size in each dimension, also called 'shape' */
  _npy_intp *dimensions;
  /*
  * Number of bytes to jump to get to the
  * next element in each dimension
  */
  _npy_intp *strides;
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
  _PyObject *base;
  /* Pointer to type structure */
  ___PyArray_Descr *descr;
  /* Flags describing array -- see below */
  int flags;
  /* For weak references */
  _PyObject *weakreflist;
} _PyArrayObject_fields;


// https://github.com/explosion/thinc/blob/master/include/numpy/__multiarray_api.h

LIBPYTHON_EXTERN void **_PyArray_API;



#define _PyArray_Type (*(_PyTypeObject *)_PyArray_API[2])


#define _PyGenericArrType_Type (*(_PyTypeObject *)_PyArray_API[10])

#define _PyArray_CastToType                                \
(*(_PyObject * (*)(_PyArrayObject *, __PyArray_Descr *, int)) \
   _PyArray_API[49])

#define _PyArray_SetBaseObject             \
 (*(int (*)(_PyArrayObject *, _PyObject *)) \
    _PyArray_API[282])

#define _PyArray_MultiplyList        \
  (*(_npy_intp (*)(_npy_intp *, int)) \
     _PyArray_API[158])                                        \

#define _PyArray_DescrFromType     \
     (*(__PyArray_Descr * (*)(int)) \
        _PyArray_API[45])

#define _PyArray_DescrFromScalar           \
      (*(__PyArray_Descr * (*)(_PyObject *)) \
         _PyArray_API[57])                                     \

#define _PyArray_CastScalarToCtype                         \
         (*(int (*)(_PyObject *, void *, __PyArray_Descr *)) \
            _PyArray_API[63])

#define _PyArray_New                                                                                          \
          (*(_PyObject * (*)(_PyTypeObject *, int, _npy_intp *, int, _npy_intp *, void *, int, int, _PyObject *)) \
             _PyArray_API[93])

inline void* _PyArray_DATA(_PyArrayObject *arr) {
  return ((_PyArrayObject_fields *)arr)->data;
}

inline _npy_intp* _PyArray_DIMS(_PyArrayObject *arr) {
  return ((_PyArrayObject_fields *)arr)->dimensions;
}

inline int _PyArray_TYPE(const _PyArrayObject *arr) {
  return ((_PyArrayObject_fields *)arr)->descr->type_num;
}

inline int _PyArray_NDIM(const _PyArrayObject *arr) {
  return ((_PyArrayObject_fields *)arr)->nd;
}

#define _PyArray_SIZE(m) _PyArray_MultiplyList(_PyArray_DIMS(m), _PyArray_NDIM(m))

#define _PyArray_Check(o) _PyObject_TypeCheck(o, &_PyArray_Type)

#define _PyArray_IsZeroDim(op) ((_PyArray_Check(op)) && \
             (_PyArray_NDIM((_PyArrayObject *)op) == 0))

#define _PyArray_IsScalar(obj, cls)                                            \
           (_PyObject_TypeCheck(obj, &_Py##cls##ArrType_Type))

#define _PyArray_CheckScalar(m) (_PyArray_IsScalar(m, Generic) ||               \
         (_PyArray_IsZeroDim(m)))                                 \


inline bool import_numpy_api(bool python3, std::string* pError) {

  _PyObject* numpy = _PyImport_ImportModule("numpy.core.multiarray");
  if (numpy == NULL) {
    *pError = "numpy.core.multiarray failed to import";
    return false;
  }

  _PyObject* c_api = _PyObject_GetAttrString(numpy, "_ARRAY_API");
  _Py_DecRef(numpy);
  if (c_api == NULL) {
    *pError = "numpy.core.multiarray _ARRAY_API not found";
    return false;
  }

  // get api pointer
  if (python3)
    _PyArray_API = (void **)_PyCapsule_GetPointer(c_api, NULL);
  else
    _PyArray_API = (void **)_PyCObject_AsVoidPtr(c_api);

  _Py_DecRef(c_api);
  if (_PyArray_API == NULL) {
    *pError = "_ARRAY_API is NULL pointer";
    return false;
  }

  return true;
}

#define _NPY_ARRAY_F_CONTIGUOUS    0x0002
#define _NPY_ARRAY_ALIGNED         0x0100
#define _NPY_ARRAY_FARRAY_RO    (_NPY_ARRAY_F_CONTIGUOUS | _NPY_ARRAY_ALIGNED)

#define _NPY_ARRAY_WRITEABLE       0x0400
#define _NPY_ARRAY_BEHAVED      (_NPY_ARRAY_ALIGNED | _NPY_ARRAY_WRITEABLE)
#define _NPY_ARRAY_FARRAY       (_NPY_ARRAY_F_CONTIGUOUS | _NPY_ARRAY_BEHAVED)

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

#endif

