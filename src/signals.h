
#ifndef RETICULATE_SIGNALS_H
#define RETICULATE_SIGNALS_H

namespace reticulate {
namespace signals {

bool getInterruptsPending();
void setInterruptsPending(bool value);

void registerInterruptHandler();

} // end namespace signals
} // end namespace reticulate

#endif /* RETICULATE_SIGNALS_H */
