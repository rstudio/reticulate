
#ifndef LIBPYTHON_TYPES_H
#define LIBPYTHON_TYPES_H

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


#ifdef LIBPYTHON_DEBUG
#define _PyObject_HEAD_EXTRA struct _object *_ob_next; struct _object *_ob_prev;
#define _PyObject_EXTRA_INIT 0, 0,
#else
#define _PyObject_HEAD_EXTRA
#define _PyObject_EXTRA_INIT
#endif


#define PyObject_HEAD   \
_PyObject_HEAD_EXTRA    \
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
PyObject_HEAD_INIT(NULL)        \
  NULL,                         \
  0,                            \
  NULL,                         \
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

typedef struct PyModuleDef {
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

} // namespace libpython

#endif /* LIBPYTHON_TYPES_H */
