
#define LIBPYTHON_DEBUG
#include "libpython-types.h"
#include "rpycall-common.h"

namespace rpycall {

void* rpycall_module_debug() {
  return &rpycallmodule;
}

void* rpycall_methods_debug() {
  return &rpycallmethods;
}

} // namespace libpython

