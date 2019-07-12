
#define LIBPYTHON_RELEASE
#include "libpython-types.h"
#include "rpycall-common.h"

namespace rpycall {

void* rpycall_module_release() {
  return &rpycallmodule;
}

void* rpycall_methods_release() {
  return &rpycallmethods;
}

} // namespace libpython

