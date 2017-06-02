// This code implements the ability to interrupt executing Python code using 
// the standard R interrupt handling mechanism.
// 
// Since execution of Python code occurs within native C++ code the normal R 
// interpeter checking for interrupts that occurs during do_eval does not have 
// a chance to execute. Typically within Rcpp packages long running user code 
// will call Rcpp::checkForInterrupts periodically to check for an interrupt 
// and exit via an exception if there is one. However there is no opportunity 
// to do this during Python execution, so we need to find a way to get periodic
// callbacks during the Python interpeter's processing to perform this check.
// 
// This is provided for via the Py_AddPendingCall function, which enables the 
// scheduling of a C callabck on the main interpeter thread *during* the 
// execution of Python code. Unfortunately this call occurs very eagerly, so we
// can't just schedule a callback and have the callback reschedule itself (this
// completely swamps the interpeteter). Rather, we need to have a background 
// thread that will perform the Py_AddPendingCall on a throttled basis (both 
// time wise and in terms of only scheduling an additional callback while the 
// Python interpeter remains running).
// 
// Having arranged for callbacks during Python interpretation at the 
// appropriate rate, we then need to interact with R to check for a user 
// interrupt, then interact with Python to notify it of any interrupt. We poll 
// for interrupts in R using the same technique as Rcpp::checkForException. If 
// we determine that an interrupt has occurred we call PyErr_SetInterrupt, 
// which signals Python that an interrupt has been requested, which will 
// ultimately result in a KeyboardInterrupt error being raised.


#include "interrupt.h"

#include "libpython.h"
using namespace libpython;

#include "tinythread.h"
using namespace tthread;

#include <Rinternals.h>

namespace {

// Class that is used to signal the need to poll for interrupts between
// threads. The function called by the Python interpreter during execution
// (pollForInterrupts) always calls requestPolling to keep polling alive. The
// background thread periodically attmeps to "collect" this request and if
// successful re-schedules the pollForInterrupts function using
// Py_AddPendingCall. This allows us to prevent the background thread from
// continually scheduling pollForInterrupts even when the Python interpreter is
// not running (because once pollForInterrupts is no longer being called by the
// Python interpreter no additonal calls to pollForInterrupts will be
// scheduled)
class InterruptPollingSignal {
public:
  InterruptPollingSignal() : pollingRequested_(true) {}
  
  void requestPolling() {
    lock_guard<mutex> lock(mutex_);
    pollingRequested_ = true;
  }
  
  bool collectRequest() {
    lock_guard<mutex> lock(mutex_);
    bool requested = pollingRequested_;
    pollingRequested_ = false;
    return requested;
  }
 
private:
  InterruptPollingSignal(const InterruptPollingSignal& other); 
  InterruptPollingSignal& operator=(const InterruptPollingSignal&);
private:
  mutex mutex_; 
  bool pollingRequested_;
};

InterruptPollingSignal s_pollingSignal;

extern "C" {

// Forward declarations
int pollForInterrupts(void*);
void checkUserInterrupt(void*);
  
// Background thread which re-schedules pollForInterrupts on the main Python
// interpreter thread every 100ms so long as the Python interpeter is still
// running (when it stops running it will stop calling pollForInterrupts and
// the polling signal will not be set).
void interruptPollingWorker(void *) {
  while(true) {
    
    // Throttle via sleep
    this_thread::sleep_for(chrono::milliseconds(500));
    
    // Schedule polling on the main thread if the interpeter is still running
    // Note that Py_AddPendingCall is documented to be callable from a background
    // thread: "This function doesn’t need a current thread state to run, and it 
    // doesn’t need the global interpreter lock."
    // (see: https://docs.python.org/3/c-api/init.html#c.Py_AddPendingCall)
    if (s_pollingSignal.collectRequest())
      Py_AddPendingCall(pollForInterrupts, NULL);
    
  }
}

// Callback function scheduled to run on the main Python interpreter loop. This
// is scheduled using Py_AddPendingCall, which ensures that it is run on the
// main thread while the interpreter is executing. Note that we can't just have
// this function keep reschedulding itself or the interpreter would be swamped
// with just calling and re-calling this function. Rather, we need to throttle
// the scheduling of the function by using a background thread + a sleep timer.
int pollForInterrupts(void*) {
  
  // Check whether an interrupt has been requested by the user. If one
  // has then set the Python interrupt flag (which will soon after result
  // in a KeyboardInterrupt error being thrown).
  if (R_ToplevelExec(checkUserInterrupt, NULL) == FALSE)
    PyErr_SetInterrupt();
  
  // Request that the background thread schedule us to be called again
  // (this is delegated to a background thread so that these requests
  // can be throttled)
  s_pollingSignal.requestPolling();
  
  // Success
  return 0;
}


// Wrapper for calling R_CheckUserInterrupt within R_TolevelExec. Note that 
// this call will result in R calling it's internal R_ProcessEvents function 
// which will allow front-ends to pump events, set the interrupt pending flag, 
// etc. This function may also jump_to_top, but these jumps are caught by 
// R_ToplevelExec and results in a return value of FALSE.
void checkUserInterrupt(void*) {
  R_CheckUserInterrupt();
}
  
} // extern "C"
} // anonymous namespace


// Initialize interrupt polling background thread
void initialize_interrupt_polling() {
  thread t(interruptPollingWorker, NULL);
  t.detach();
}






