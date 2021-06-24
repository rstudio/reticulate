
#include "signals.h"

#include "reticulate.h"

#ifndef _WIN32
# include <string.h>
# include <signal.h>
#else
# include <Windows.h>
#endif

#include "libpython.h"
using namespace libpython;

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

// flag indicating whether a callback is registered
volatile sig_atomic_t s_interruptCallbackRegistered;

// callback added via Py_AddPendingCall which handles R interrupts
int pyInterruptCallback(void* data)
{
  s_interruptCallbackRegistered = 0;
  
  DBG("Invoking interrupt callback.\n");
  
  // if an R interrupt was signaled, but the Python interpreter
  // got to it first, then let Python handle the interrupt instead
  if (getInterruptsPending())
  {
    DBG("Interrupting Python!\n");
    
    setInterruptsPending(false);
    PyErr_SetInterrupt();
  }
  
  
  return 0;
}

} // end anonymous namespace

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

void interruptHandler(int signum) {
  
  DBG("Invoking interrupt handler.\n");
  
  // set R interrupts pending
  setInterruptsPending(true);
  
  // add pending call (to be used to handle interrupt if appropriate)
  if (s_interruptCallbackRegistered == 0)
  {
    s_interruptCallbackRegistered = 1;
    Py_AddPendingCall(pyInterruptCallback, NULL);
  }
  
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

BOOL CALLBACK consoleCtrlHandler(DWORD type)
{
  switch (type)
  {
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
