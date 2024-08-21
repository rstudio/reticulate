#ifndef PENDING_PY_CALLS_NOTIFIER_H
#define PENDING_PY_CALLS_NOTIFIER_H

#include <functional>
#include <atomic>
#include "tinythread.h"

namespace pending_py_calls_notifier {

    // Initialize the notifier with a function that runs pending calls.
    void initialize(std::function<void()> run_pending_calls);

    // Notify the main thread to run pending calls.
    void notify();

    // Undo initialize
    void deinitialize();

} // namespace pending_py_calls_notifier

#endif // PENDING_PY_CALLS_NOTIFIER_H
