
#include <Rcpp.h>
using namespace Rcpp;

#include "libpython-types.h"
#include "reticulate_types.h"

#include "event_loop.h"
#include "tinythread.h"

#include <fstream>
#include <time.h>

using namespace libpython;

namespace libpython {

extern "C" PyObject* call_r_function(PyObject *self, PyObject* args, PyObject* keywords);
extern "C" PyObject* call_python_function_on_main_thread(PyObject *self, PyObject* args, PyObject* keywords);

} // end namespace libpython

namespace rpycall {
namespace {

PyMethodDef rpycallmethods[] = {

  {
    "call_r_function", (PyCFunction)call_r_function,
    METH_VARARGS | METH_KEYWORDS, "Call an R function"
  },

  {
    "call_python_function_on_main_thread", (PyCFunction)call_python_function_on_main_thread,
    METH_VARARGS | METH_KEYWORDS, "Call a Python function on the main thread"
  },

  {
    NULL, NULL,
    0, NULL
  }

};

static struct PyModuleDef rpycallmodule = {
  PyModuleDef_HEAD_INIT,
  "rpycall",
  NULL,
  -1,
  rpycallmethods,
  NULL,
  NULL,
  NULL,
  NULL
};

} // end namespace rpycall
} // end anonymous namespace

