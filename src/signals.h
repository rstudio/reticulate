
#ifndef RETICULATE_SIGNALS_H
#define RETICULATE_SIGNALS_H

namespace reticulate {
namespace signals {

// Python interrupt state
bool getPythonInterruptsPending();
void setPythonInterruptsPending(bool value);

// R interrupts state
bool getRInterruptsPending();
void setRInterruptsPending(bool value);


} // end namespace signals
} // end namespace reticulate

#endif /* RETICULATE_SIGNALS_H */
