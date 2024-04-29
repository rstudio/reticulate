
#include "signals.h"
#include "common.h"

#ifndef _WIN32
# include <string.h>
# include <signal.h>
#else
# include <windows.h>
#endif

#include "libpython.h"
using namespace reticulate::libpython;

extern "C" {

// flag indicating whether R interrupts are pending
#ifndef _WIN32
extern int R_interrupts_pending;
#else
LibExtern int UserBreak;
#endif

// flag indicating if interrupts are suspended
// note that R doesn't use this on Windows when checking
// for interrupts in R_ProcessEvents
LibExtern Rboolean R_interrupts_suspended;

}

namespace reticulate {
namespace signals {


bool getInterruptsPending() {
#ifndef _WIN32
  return R_interrupts_pending != 0;
#else
  return UserBreak != 0;
#endif
}

void setInterruptsPending(bool value) {

#ifndef _WIN32
  R_interrupts_pending = value ? 1 : 0;
#else
  UserBreak = value ? 1 : 0;
#endif

}

bool getInterruptsSuspended() {
  return R_interrupts_suspended != 0;
}

void setInterruptsSuspended(bool value) {
  R_interrupts_suspended = value ? TRUE : FALSE;
}

} // end namespace signals
} // end namespace reticulate

