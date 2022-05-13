
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
LibExtern int R_interrupts_suspended;

}

namespace reticulate {
namespace signals {

namespace {
volatile sig_atomic_t s_pyInterruptsPending;
} // end anonymous namespace

bool getPythonInterruptsPending() {
  return s_pyInterruptsPending != 0;
}

void setPythonInterruptsPending(bool value) {
  s_pyInterruptsPending = value ? 1 : 0;
}

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
  R_interrupts_suspended = value ? 1 : 0;
}

// namespace {
//
// volatile sig_atomic_t s_pyInterruptHandlerRegistered;
//
// int pyInterruptHandler(void*) {
//
//   DBG("Running Python interrupt handler.");
//   s_pyInterruptHandlerRegistered = 0;
//
//   if (s_pyInterruptsPending != 0)
//   {
//     DBG("Python interrupts are pending; setting interrupt.");
//     s_pyInterruptsPending = 0;
//     PyErr_SetInterrupt();
//   }
//   else
//   {
//     DBG("No Python interrupts pending.");
//   }
//
//   return 0;
// }
//
// } // end anonymous namespace

void interruptHandler(int signum) {

  DBG("Invoking interrupt handler.");

  // set internal flag for Python interrupts
  setPythonInterruptsPending(true);

  // set R interrupts pending
  setInterruptsPending(true);

  // tell Python to interrupt
  PyErr_SetInterrupt();

}

#ifndef _WIN32

void registerInterruptHandlerUnix() {

  // initialize sigaction struct
  struct sigaction sigint;
  memset(&sigint, 0, sizeof sigint);
  sigemptyset(&sigint.sa_mask);
  sigint.sa_flags = 0;

  // set handler
  sigint.sa_handler = interruptHandler;

  // install signal handler
  sigaction(SIGINT, &sigint, NULL);

}

#else

BOOL CALLBACK consoleCtrlHandler(DWORD type) {
  switch (type) {
  case CTRL_C_EVENT:
  case CTRL_BREAK_EVENT:
    interruptHandler(2);
    return TRUE;
  default:
    return FALSE;
  }
}

void registerInterruptHandlerWin32() {
  ::SetConsoleCtrlHandler(consoleCtrlHandler, TRUE);
}

#endif

void registerInterruptHandler() {
#ifndef _WIN32
  registerInterruptHandlerUnix();
#else
  registerInterruptHandlerWin32();
#endif
}

} // end namespace signals
} // end namespace reticulate

// [[Rcpp::export]]
void py_register_interrupt_handler() {
  reticulate::signals::registerInterruptHandler();
}


// [[Rcpp::export]]
bool py_interrupts_pending(bool reset) {

  if (reticulate::signals::getInterruptsSuspended())
    return false;

  if (reset) {
    reticulate::signals::setInterruptsPending(false);
    return false;
  }

  return reticulate::signals::getInterruptsPending();

}
