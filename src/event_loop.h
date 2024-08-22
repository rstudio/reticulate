
#ifndef RETICULATE_EVENT_LOOP_H
#define RETICULATE_EVENT_LOOP_H

namespace reticulate {
namespace event_loop {

void initialize();

void deinitialize(bool wait = false);

} // namespace event_loop
} // namespace reticulate

#endif // RETICULATE_EVENT_LOOP_H
