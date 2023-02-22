
#ifndef RETICULATE_LIBPYTHON_H
#define RETICULATE_LIBPYTHON_H

#include <string>
#include <ostream>
#include <stdint.h>

#ifndef LIBPYTHON_CPP
#define LIBPYTHON_EXTERN extern
#else
#define LIBPYTHON_EXTERN
#endif

#define _PYTHON_API_VERSION 1013
#define _PYTHON3_ABI_VERSION 3

namespace reticulate {
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
#define Py_eval_input 258

#ifdef RETICULATE_PYTHON_DEBUG

#define _PyObject_HEAD_EXTRA            \
    struct _object *_ob_next;           \
    struct _object *_ob_prev;

#define _PyObject_EXTRA_INIT 0, 0,

#else

#define _PyObject_HEAD_EXTRA
#define _PyObject_EXTRA_INIT

#endif /* RETICULATE_PYTHON_DEBUG */

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

typedef struct PyCompilerFlags{
  int cf_flags;
  int cf_feature_version;
} PyCompilerFlags;

typedef Py_ssize_t Py_hash_t;

LIBPYTHON_EXTERN PyTypeObject* PyFunction_Type;
LIBPYTHON_EXTERN PyTypeObject* PyModule_Type;
LIBPYTHON_EXTERN PyTypeObject* PyType_Type;
LIBPYTHON_EXTERN PyTypeObject* PyProperty_Type;
LIBPYTHON_EXTERN PyTypeObject* PyMethod_Type;
LIBPYTHON_EXTERN PyTypeObject *PyMethod_Type;

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
LIBPYTHON_EXTERN PyObject* Py_ByteArray;
LIBPYTHON_EXTERN PyObject* PyExc_KeyboardInterrupt;

void initialize_type_objects(bool python3);

#define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)

#define PyUnicode_Check(o)   (Py_TYPE(o) == Py_TYPE(Py_Unicode))
#define PyString_Check(o)    (Py_TYPE(o) == Py_TYPE(Py_String))
#define PyInt_Check(o)       (Py_TYPE(o) == Py_TYPE(Py_Int))
#define PyLong_Check(o)      (Py_TYPE(o) == Py_TYPE(Py_Long))
#define PyDict_Check(o)      (Py_TYPE(o) == Py_TYPE(Py_Dict))
#define PyFloat_Check(o)     (Py_TYPE(o) == Py_TYPE(Py_Float))
#define PyTuple_Check(o)     (Py_TYPE(o) == Py_TYPE(Py_Tuple))
#define PyList_Check(o)      (Py_TYPE(o) == Py_TYPE(Py_List))
#define PyComplex_Check(o)   (Py_TYPE(o) == Py_TYPE(Py_Complex))
#define PyByteArray_Check(o) (Py_TYPE(o) == Py_TYPE(Py_ByteArray))

#define PyBool_Check(o)      ((o == Py_False) || (o == Py_True))
#define PyFunction_Check(op) ((PyTypeObject*)(Py_TYPE(op)) == PyFunction_Type)
#define PyMethod_Check(op)   ((PyTypeObject *)(Py_TYPE(op)) == PyMethod_Type)

LIBPYTHON_EXTERN void (*Py_Initialize)();
LIBPYTHON_EXTERN int (*Py_IsInitialized)();
LIBPYTHON_EXTERN const char* (*Py_GetVersion)();
LIBPYTHON_EXTERN char* (*Py_GetProgramFullPath_v2)();
LIBPYTHON_EXTERN wchar_t* (*Py_GetProgramFullPath)();


LIBPYTHON_EXTERN int (*Py_AddPendingCall)(int (*func)(void *), void *arg);
LIBPYTHON_EXTERN void (*PyErr_SetInterrupt)();
LIBPYTHON_EXTERN void (*PyErr_CheckSignals)();

LIBPYTHON_EXTERN PyObject* (*Py_InitModule4)(const char *name, PyMethodDef *methods,
           const char *doc, PyObject *self,
           int apiver);

LIBPYTHON_EXTERN PyObject* (*PyImport_ImportModule)(const char *name);
LIBPYTHON_EXTERN PyObject* (*PyImport_Import)(PyObject * name);
LIBPYTHON_EXTERN PyObject* (*PyImport_GetModuleDict)();


LIBPYTHON_EXTERN PyObject* (*PyModule_Create)(PyModuleDef *def, int);
LIBPYTHON_EXTERN int (*PyImport_AppendInittab)(const char *name, PyObject* (*initfunc)());

LIBPYTHON_EXTERN PyObject* (*Py_BuildValue)(const char *format, ...);

LIBPYTHON_EXTERN void (*Py_IncRef)(PyObject *);
LIBPYTHON_EXTERN void (*Py_DecRef)(PyObject *);

LIBPYTHON_EXTERN int (*PyObject_Print)(PyObject* o, FILE* fp, int flags);
LIBPYTHON_EXTERN PyObject* (*PyObject_Str)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyObject_Repr)(PyObject *);

LIBPYTHON_EXTERN int (*PyObject_IsInstance)(PyObject *object, PyObject *typeorclass);

/* Rich comparison opcodes */
#define Py_LT 0
#define Py_LE 1
#define Py_EQ 2
#define Py_NE 3
#define Py_GT 4
#define Py_GE 5
LIBPYTHON_EXTERN int (*PyObject_RichCompareBool)(PyObject *o1, PyObject *o2, int opid);

LIBPYTHON_EXTERN PyObject* (*PyObject_Dir)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*PyObject_Call)(PyObject *callable_object,
           PyObject *args, PyObject *kw);
LIBPYTHON_EXTERN PyObject* (*PyObject_CallFunctionObjArgs)(PyObject *callable,
           ...);

LIBPYTHON_EXTERN Py_ssize_t (*PyObject_Size)(PyObject*);
LIBPYTHON_EXTERN PyObject* (*PyObject_GetAttr)(PyObject*, PyObject*);
LIBPYTHON_EXTERN int (*PyObject_HasAttr)(PyObject*, PyObject*);
LIBPYTHON_EXTERN int (*PyObject_SetAttr)(PyObject*, PyObject*, PyObject*);

LIBPYTHON_EXTERN PyObject* (*PyObject_GetAttrString)(PyObject*, const char *);
LIBPYTHON_EXTERN int (*PyObject_HasAttrString)(PyObject*, const char *);
LIBPYTHON_EXTERN int (*PyObject_SetAttrString)(PyObject*, const char *, PyObject*);

LIBPYTHON_EXTERN PyObject* (*PyObject_GetItem)(PyObject*, PyObject*);
LIBPYTHON_EXTERN int (*PyObject_SetItem)(PyObject*, PyObject*, PyObject*);
LIBPYTHON_EXTERN int (*PyObject_DelItem)(PyObject*, PyObject*);

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
    PyObject *obj,	/* string or Unicode object */
    char **s,		/* pointer to buffer variable */
    Py_ssize_t *len	/* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);

LIBPYTHON_EXTERN PyObject* (*PyString_FromString)(const char *);
LIBPYTHON_EXTERN PyObject* (*PyString_FromStringAndSize)(const char *, Py_ssize_t);

LIBPYTHON_EXTERN PyObject* (*PyUnicode_EncodeLocale)(PyObject *unicode, const char *errors);
LIBPYTHON_EXTERN PyObject* (*PyUnicode_AsEncodedString)(PyObject *unicode, const char *encoding, const char *errors);
LIBPYTHON_EXTERN int (*PyBytes_AsStringAndSize)(
    PyObject *obj,      /* string or Unicode object */
    char **s,           /* pointer to buffer variable */
    Py_ssize_t *len     /* pointer to length variable or NULL
  (only possible for 0-terminated
  strings) */
);
#ifdef _WIN32
LIBPYTHON_EXTERN PyObject* (*PyUnicode_AsMBCSString)(PyObject *unicode);
#endif

LIBPYTHON_EXTERN PyObject* (*PyBytes_FromStringAndSize)(const char *, Py_ssize_t);
LIBPYTHON_EXTERN Py_ssize_t (*PyByteArray_Size)(PyObject *bytearray);
LIBPYTHON_EXTERN PyObject* (*PyByteArray_FromStringAndSize)(const char *string, Py_ssize_t len);
LIBPYTHON_EXTERN char* (*PyByteArray_AsString)(PyObject *bytearray);
LIBPYTHON_EXTERN PyObject* (*PyUnicode_FromString)(const char *u);

LIBPYTHON_EXTERN void (*PyErr_Clear)();
LIBPYTHON_EXTERN void (*PyErr_Fetch)(PyObject **, PyObject **, PyObject **);
LIBPYTHON_EXTERN void (*PyErr_Restore)(PyObject *, PyObject *, PyObject *);
LIBPYTHON_EXTERN void (*PyErr_SetNone)(PyObject*);
LIBPYTHON_EXTERN void (*PyErr_BadArgument)();
LIBPYTHON_EXTERN PyObject* (*PyErr_Occurred)(void);
LIBPYTHON_EXTERN void (*PyErr_NormalizeException)(PyObject**, PyObject**, PyObject**);
LIBPYTHON_EXTERN int (*PyErr_GivenExceptionMatches)(PyObject *given, PyObject *exc);
LIBPYTHON_EXTERN int (*PyErr_ExceptionMatches)(PyObject *exc);
LIBPYTHON_EXTERN int (*PyException_SetTraceback)(PyObject *ex, PyObject *tb);

LIBPYTHON_EXTERN int (*PyCallable_Check)(PyObject *);

LIBPYTHON_EXTERN PyObject* (*PyModule_GetDict)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyImport_AddModule)(const char *);

LIBPYTHON_EXTERN PyObject* (*PyRun_FileEx)(FILE*, const char*, int, PyObject*, PyObject*, int);
LIBPYTHON_EXTERN PyObject* (*PyRun_StringFlags)(const char *, int, PyObject*, PyObject*, void*);
LIBPYTHON_EXTERN PyObject* (*Py_CompileString)(const char *str, const char *filename, int start);
LIBPYTHON_EXTERN PyObject* (*PyEval_EvalCode)(PyObject *co, PyObject *globals, PyObject *locals);

LIBPYTHON_EXTERN PyObject* (*PyObject_GetIter)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyIter_Next)(PyObject *);

typedef void (*PyCapsule_Destructor)(PyObject *);
LIBPYTHON_EXTERN PyObject* (*PyCapsule_New)(void *pointer, const char *name, PyCapsule_Destructor destructor);
LIBPYTHON_EXTERN void* (*PyCapsule_GetPointer)(PyObject *capsule, const char *name);
LIBPYTHON_EXTERN void* (*PyCapsule_GetContext)(PyObject *capsule);
LIBPYTHON_EXTERN int (*PyCapsule_SetContext)(PyObject *capsule, void *context);
LIBPYTHON_EXTERN int (*PyCapsule_IsValid)(PyObject *capsule, const char *name);


LIBPYTHON_EXTERN PyObject* (*PyDict_New)(void);
LIBPYTHON_EXTERN int (*PyDict_Contains)(PyObject *mp, PyObject *key);
LIBPYTHON_EXTERN PyObject* (*PyDict_GetItem)(PyObject *mp, PyObject *key);
LIBPYTHON_EXTERN int (*PyDict_SetItem)(PyObject *mp, PyObject *key, PyObject *item);
LIBPYTHON_EXTERN int (*PyDict_SetItemString)(PyObject *dp, const char *key, PyObject *item);
LIBPYTHON_EXTERN int (*PyDict_DelItemString)(PyObject *dp, const char *key);
LIBPYTHON_EXTERN int (*PyDict_Next)(
    PyObject *mp, Py_ssize_t *pos, PyObject **key, PyObject **value);
LIBPYTHON_EXTERN PyObject* (*PyDict_Keys)(PyObject *mp);
LIBPYTHON_EXTERN PyObject* (*PyDict_Values)(PyObject *mp);
LIBPYTHON_EXTERN Py_ssize_t (*PyDict_Size)(PyObject *mp);
LIBPYTHON_EXTERN PyObject* (*PyDict_Copy)(PyObject *mp);

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

LIBPYTHON_EXTERN void (*Py_SetProgramName)(char *);
LIBPYTHON_EXTERN void (*Py_SetProgramName_v3)(wchar_t *);

LIBPYTHON_EXTERN void (*Py_SetPythonHome)(char *);
LIBPYTHON_EXTERN void (*Py_SetPythonHome_v3)(wchar_t *);

LIBPYTHON_EXTERN void (*PySys_SetArgv)(int, char **);
LIBPYTHON_EXTERN void (*PySys_SetArgv_v3)(int, wchar_t **);

LIBPYTHON_EXTERN void (*PySys_WriteStderr)(const char *format, ...);
LIBPYTHON_EXTERN PyObject* (*PySys_GetObject)(const char *name);

LIBPYTHON_EXTERN PyObject* (*PyObject_CallMethod)(PyObject *o, const char *name, const char *format, ...);
LIBPYTHON_EXTERN PyObject* (*PySequence_GetItem)(PyObject *o, Py_ssize_t i);
LIBPYTHON_EXTERN int (*PyObject_IsTrue)(PyObject *o);

LIBPYTHON_EXTERN PyObject* (*Py_CompileStringExFlags)(const char *str, const char *filename, int start, PyCompilerFlags *flags, int optimize);

LIBPYTHON_EXTERN void* (*PyCapsule_Import)(const char *name, int no_block);

LIBPYTHON_EXTERN PyObject* (*PyObject_Type)(PyObject* o);
#define PyObject_TypeCheck(o, tp) ((PyTypeObject*)Py_TYPE(o) == (tp)) || PyType_IsSubtype((PyTypeObject*)Py_TYPE(o), (tp))

#define PyType_Check(o) PyObject_TypeCheck(o, PyType_Type)

#define PyModule_Check(op) PyObject_TypeCheck(op, PyModule_Type)
#define PyModule_CheckExact(op) (Py_TYPE(op) == PyModule_Type)


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


// has not changed in 6 years, if it changes then it implies that our PyArray_API
// indexes may be off
// see: https://github.com/numpy/numpy/blame/master/numpy/core/setup_common.py#L26
#define NPY_VERSION 0x01000009

// checks for numpy 1.6 / 1.7
// see: https://github.com/numpy/numpy/blob/master/numpy/core/code_generators/cversions.txt
#define NPY_1_6_API_VERSION 0x00000006
#define NPY_1_7_API_VERSION 0x00000007

#define PyArray_GetNDArrayCVersion (*(unsigned int (*)(void)) PyArray_API[0])

#define PyArray_GetNDArrayCFeatureVersion (*(unsigned int (*)(void)) PyArray_API[211])

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

// NOTE: PyArray_Malloc has a bit of indirection for different
// versions of Python, but the underlying intention is to
// always use the system allocator (through malloc). See:
//
// https://github.com/numpy/numpy/blob/f5758d6fe15c2b506290bfc5379a10027617b331/numpy/core/include/numpy/ndarraytypes.h#L348-L367
// https://github.com/python/cpython/blob/28f713601d3ec80820e842dcb25a234093f1ff18/Objects/obmalloc.c#L66-L76
//
#ifdef __cplusplus
#define PyArray_malloc std::malloc
#else
#define PyArray_malloc malloc
#endif

#define PyArray_New                                                                                          \
          (*(PyObject * (*)(PyTypeObject *, int, npy_intp *, int, npy_intp *, void *, int, int, PyObject *)) \
             PyArray_API[93])

inline void* PyArray_DATA(PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->data;
}

#define PyArray_BASE(arr) (((PyArrayObject_fields *)(arr))->base)

inline npy_intp* PyArray_DIMS(PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->dimensions;
}

inline int PyArray_TYPE(const PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->descr->type_num;
}

inline int PyArray_NDIM(const PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->nd;
}

inline int PyArray_FLAGS(PyArrayObject *arr) {
  return ((PyArrayObject_fields *)arr)->flags;
}

#define PyArray_SIZE(m) PyArray_MultiplyList(PyArray_DIMS(m), PyArray_NDIM(m))

#define PyArray_Check(o) PyObject_TypeCheck(o, &PyArray_Type)

#define PyArray_IsZeroDim(op) ((PyArray_Check(op)) && \
             (PyArray_NDIM((PyArrayObject *)op) == 0))

#define PyArray_IsScalar(obj, cls)                                            \
           (PyObject_TypeCheck(obj, &Py##cls##ArrType_Type))

#define PyArray_CheckScalar(m) (PyArray_IsScalar(m, Generic) ||               \
         (PyArray_IsZeroDim(m)))                                 \


bool import_numpy_api(bool python3, std::string* pError);

#define NPY_ARRAY_C_CONTIGUOUS    0x0001
#define NPY_ARRAY_F_CONTIGUOUS    0x0002
#define NPY_ARRAY_OWNDATA         0x0004
#define NPY_ARRAY_ALIGNED         0x0100
#define NPY_ARRAY_FARRAY_RO    (NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED)
#define NPY_ARRAY_CARRAY_RO    (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)

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

/* Start PyFrameObject */

/* Bytecode object */
typedef struct {
  PyObject_HEAD
  int co_argcount;		/* #arguments, except *args */
  int co_nlocals;		/* #local variables */
  int co_stacksize;		/* #entries needed for evaluation stack */
  int co_flags;		/* CO_..., see below */
  PyObject *co_code;		/* instruction opcodes */
  PyObject *co_consts;	/* list (constants used) */
  PyObject *co_names;		/* list of strings (names used) */
  PyObject *co_varnames;	/* tuple of strings (local variable names) */
  PyObject *co_freevars;	/* tuple of strings (free variable names) */
  PyObject *co_cellvars;      /* tuple of strings (cell variable names) */
  /* The rest doesn't count for hash/cmp */
  PyObject *co_filename;	/* string (where it was loaded from) */
  PyObject *co_name;		/* string (name, for reference) */
  int co_firstlineno;		/* first source line number */
  PyObject *co_lnotab;	/* string (encoding addr<->lineno mapping) See
  Objects/lnotab_notes.txt for details. */
  void *co_zombieframe;     /* for optimization only (see frameobject.c) */
  PyObject *co_weakreflist;   /* to support weakrefs to code objects */
} PyCodeObject;

#define CO_MAXBLOCKS 20
typedef struct {
  int b_type;         /* what kind of block this is */
  int b_handler;      /* where to jump to find handler */
  int b_level;        /* value stack level to pop to */
} PyTryBlock;

typedef struct _is {

  struct _is *next;
  struct _ts *tstate_head;

  PyObject *modules;
  PyObject *modules_by_index;
  PyObject *sysdict;
  PyObject *builtins;
  PyObject *modules_reloading;

  PyObject *codec_search_path;
  PyObject *codec_search_cache;
  PyObject *codec_error_registry;
  int codecs_initialized;

} PyInterpreterState;

typedef int (*Py_tracefunc)(PyObject *, struct _frame *, int, PyObject *);

typedef struct _ts {
  /* See Python/ceval.c for comments explaining most fields */

  struct _ts *next;
  PyInterpreterState *interp;

  struct _frame *frame;
  int recursion_depth;
  char overflowed; /* The stack has overflowed. Allow 50 more calls
  to handle the runtime error. */
  char recursion_critical; /* The current calls must not cause
  a stack overflow. */
  /* 'tracing' keeps track of the execution depth when tracing/profiling.
  This is to prevent the actual trace/profile code from being recorded in
  the trace/profile. */
  int tracing;
  int use_tracing;

  Py_tracefunc c_profilefunc;
  Py_tracefunc c_tracefunc;
  PyObject *c_profileobj;
  PyObject *c_traceobj;

  PyObject *curexc_type;
  PyObject *curexc_value;
  PyObject *curexc_traceback;

  PyObject *exc_type;
  PyObject *exc_value;
  PyObject *exc_traceback;

  PyObject *dict;

  int tick_counter;

  int gilstate_counter;

  PyObject *async_exc;
  long thread_id;

} PyThreadState;

typedef struct _frame {
  PyObject_VAR_HEAD
  struct _frame *f_back;	/* previous frame, or NULL */
  PyCodeObject *f_code;	/* code segment */
  PyObject *f_builtins;	/* builtin symbol table (PyDictObject) */
  PyObject *f_globals;	/* global symbol table (PyDictObject) */
  PyObject *f_locals;		/* local symbol table (any mapping) */
  PyObject **f_valuestack;	/* points after the last local */
  /* Next free slot in f_valuestack.  Frame creation sets to f_valuestack.
  Frame evaluation usually NULLs it, but a frame that yields sets it
  to the current stack top. */
  PyObject **f_stacktop;
  PyObject *f_trace;		/* Trace function */

  PyObject *f_exc_type, *f_exc_value, *f_exc_traceback;

  PyThreadState *f_tstate;
  int f_lasti;		/* Last instruction if called */

  int f_lineno;		/* Current line number */
  int f_iblock;		/* index in f_blockstack */
  PyTryBlock f_blockstack[CO_MAXBLOCKS]; /* for try and loop blocks */
  PyObject *f_localsplus[1];	/* locals+stack, dynamically sized */
} PyFrameObject;

typedef
  enum {PyGILState_LOCKED, PyGILState_UNLOCKED}
PyGILState_STATE;

LIBPYTHON_EXTERN void (*PyEval_SetProfile)(Py_tracefunc func, PyObject *obj);
LIBPYTHON_EXTERN PyThreadState* (*PyGILState_GetThisThreadState)(void);
LIBPYTHON_EXTERN PyGILState_STATE (*PyGILState_Ensure)(void);
LIBPYTHON_EXTERN void (*PyGILState_Release)(PyGILState_STATE);
LIBPYTHON_EXTERN PyThreadState* (*PyThreadState_Next)(PyThreadState*);

/* End PyFrameObject */

} // namespace libpython
} // namespace reticulate

#endif /* RETICULATE_LIBPYTHON_H */
