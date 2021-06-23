
#include "signals.h"

#ifndef _WIN32
# include <string.h>
# include <signal.h>
#else
# include <Windows.h>
#endif

#include "libpython.h"
using namespace libpython;

// flag indicating whether R interrupts are pending
extern "C" {
#ifndef _WIN32
extern int R_interrupts_pending;
#else
extern __declspec(dllimport) int UserBreak;
#endif
}

namespace reticulate {
namespace signals {

namespace {

// callback added via Py_AddPendingCall which handles R interrupts
int pyInterruptCallback(void* data)
{
  // if an R interrupt was signaled, but the Python interpreter
  // got to it first, then let Python handle the interrupt instead
  if (getInterruptsPending())
  {
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

void interruptHandler(int signum) {
  
  // set R interrupts pending
  setInterruptsPending(true);
  
  // add pending call (to be used to handle interrupt if appropriate)
  PyGILState_STATE state = PyGILState_Ensure();
  Py_AddPendingCall(pyInterruptCallback, NULL);
  PyGILState_Release(state);
  
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
    return true;
  default:
    return false;
  }
}

void registerInterruptHandlerWin32() {

  // accept Ctrl + C interrupts
  ::SetConsoleCtrlHandler(NULL, FALSE);

  // remove an old registration, if any
  ::SetConsoleCtrlHandler(consoleCtrlHandler, FALSE);

  // and register the handler
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
