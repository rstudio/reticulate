
#ifndef _WIN32
# include <string.h>
# include <signal.h>
#else
# include <Windows.h>
#endif

namespace libpython {
extern void (*PyErr_SetInterrupt)();
} // namespace libpython

// flag indicating whether R interrupts are pending
#ifndef _WIN32
extern "C" int R_interrupts_pending;
#else
extern "C" int UserBreak;
#endif

namespace reticulate {
namespace signals {

namespace {

// boolean check if the Python interrupt handler was fired
bool s_pythonInterrupted;

} // end anonymous namespace

bool getPythonInterruptsPending() {
  return s_pythonInterrupted;
}

void setPythonInterruptsPending(bool value) {
  s_pythonInterrupted = value;
}

bool getRInterruptsPending() {
#ifndef _WIN32
  return R_interrupts_pending != 0;
#else
  return UserBreak != 0;
#endif
}

void setRInterruptsPending(bool value) {
  
#ifndef _WIN32
  R_interrupts_pending = value ? 1 : 0;
#else
  UserBreak = value ? 1 : 0;
#endif
  
}

void interruptHandler(int signum) {
  
  // set R interrupts pending
  setRInterruptsPending(true);
  
  // mark internal flag
  s_pythonInterrupted = true;
  
}

} // end namespace signals
} // end namespace reticulate

// [[Rcpp::export]]
void py_interrupt_handler(int signum) {
  reticulate::signals::interruptHandler(signum);
}
