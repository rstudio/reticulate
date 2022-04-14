
#ifndef RETICULATE_SIGNALS_H
#define RETICULATE_SIGNALS_H

namespace reticulate {
namespace signals {

bool getPythonInterruptsPending();
void setPythonInterruptsPending(bool value);

bool getInterruptsPending();
void setInterruptsPending(bool value);

bool getInterruptsSuspended();
void setInterruptsSuspended(bool value);

void registerInterruptHandler();

// NOTE: We also manage the interrupts pending flag here since calls to
// R_ProcessEvents (on Windows) will check UserBreak without respecting
// the R_interrupts_suspended flag.
class InterruptsSuspendedScope
{
public:

  InterruptsSuspendedScope()
    : pending_(getInterruptsPending()),
      suspended_(getInterruptsSuspended())
  {
    setInterruptsPending(false);
    setInterruptsSuspended(true);
  }

  ~InterruptsSuspendedScope()
  {
    setInterruptsPending(pending_ || getInterruptsPending());
    setInterruptsSuspended(suspended_);
  }

private:
  int pending_;
  int suspended_;
};

} // end namespace signals
} // end namespace reticulate

#endif /* RETICULATE_SIGNALS_H */
