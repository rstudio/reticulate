#include "pending_py_calls_notifier.h"
#include <Rinternals.h>
#include <R_ext/eventloop.h> // for addInputHandler()
#include <atomic>

namespace pending_py_calls_notifier {

namespace {

std::atomic<bool> notification_pending(false);
std::function<void()> run_pending_calls_func;

#ifdef _WIN32
HWND message_window;
const UINT WM_PY_PENDING_CALLS = WM_USER + 1;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  if (uMsg == WM_PY_PENDING_CALLS) {
    if (notification_pending.exchange(false)) {
      run_pending_calls_func();
    }
    return 0;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void initialize_windows_message_window() {
  HINSTANCE hInstance = GetModuleHandle(NULL);
  WNDCLASS wc = {0};
  wc.lpfnWndProc = WindowProc;
  wc.hInstance = hInstance;
  wc.lpszClassName = TEXT("ReticulatePyPendingCallsNotifier");

  RegisterClass(&wc);
  message_window = CreateWindow(TEXT("ReticulatePyPendingCallsNotifier"), NULL, 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, hInstance, NULL);
}

#else
int pipe_fds[2]; // Pipe file descriptors for inter-thread communication
InputHandler* input_handler = nullptr;

void input_handler_function(void* userData) {
  char buffer[1];
  if (read(pipe_fds[0], buffer, 2) == -1) // Clear the pipe
    REprintf("Failed to read from pipe for pending Python calls notifier");
  if (notification_pending.exchange(false)) {
    run_pending_calls_func();
  }
}
#endif

} // anonymous namespace

void initialize(std::function<void()> run_pending_calls) {
  // Set the function for running pending Python calls
  run_pending_calls_func = run_pending_calls;

#ifdef _WIN32
  initialize_windows_message_window();
#else
  // Create a pipe for inter-thread communication (POSIX)
  if (pipe(pipe_fds) == -1) {
    Rf_error("Failed to create pipe for pending Python calls notifier");
  }

  // Add the input handler to the R event loop
  input_handler = addInputHandler(R_InputHandlers, pipe_fds[0], input_handler_function, 88); // Choose an appropriate activity ID
#endif
}

void notify() {
  if (!notification_pending.exchange(true)) {
#ifdef _WIN32
    PostMessage(message_window, WM_PY_PENDING_CALLS, 0, 0);
#else
    if (write(pipe_fds[1], "x", 1) == -1) {
      REprintf("Failed to write to pipe for pending Python calls notifier\n");
    }
#endif
  }

}

void deinitialize() {
#ifdef _WIN32
#else
  removeInputHandler(&R_InputHandlers, input_handler);
#endif
}

} // namespace pending_py_calls_notifier
