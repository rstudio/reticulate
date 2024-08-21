
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <atomic>
#include <functional>

// tinythread.h is only used for its includes. It does platform-specific includes
// nicely like <windows.h>, <unistd.h>, etc.
#include "tinythread.h"

#define R_NO_REMAP
#include <Rinternals.h>

#ifndef _WIN32
#include <R_ext/eventloop.h> // for addInputHandler(), removeInputHandler()
#endif

#include "pending_py_calls_notifier.h"

namespace pending_py_calls_notifier {

namespace {

std::atomic<bool> notification_pending(false);
std::function<void()> run_pending_calls;

}

#ifdef _WIN32

namespace {

HWND message_window;
const UINT WM_PY_PENDING_CALLS = WM_USER + 1;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  if (uMsg == WM_PY_PENDING_CALLS) {
    if (notification_pending.exchange(false)) {
      run_pending_calls();
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

} // end anonymous namespace, windows-specific


void initialize(std::function<void()> run_pending_calls_func) {
  initialize_windows_message_window();
}
void notify() {
    PostMessage(message_window, WM_PY_PENDING_CALLS, 0, 0);
}

void deinitialize() {
  if (message_window) {
    DestroyWindow(message_window);
    message_window = nullptr;
  }

#else // end windows, start unix

namespace {

int pipe_fds[2]; // Pipe file descriptors for inter-thread communication
InputHandler* input_handler = nullptr;

void input_handler_function(void* userData) {
  char buffer[4];
  if (read(pipe_fds[0], buffer, sizeof(buffer)) == -1) // Clear the pipe
    REprintf("Failed to read from pipe for pending Python calls notifier");
  if (notification_pending.exchange(false)) {
    run_pending_calls();
  }
}

} // end anonymous namespace, unix-specific


void initialize(std::function<void()> run_pending_calls_func) {
  run_pending_calls = run_pending_calls_func;
  if (pipe(pipe_fds) == -1)
    Rf_error("Failed to create pipe for pending Python calls notifier");

  input_handler = addInputHandler(R_InputHandlers, pipe_fds[0], input_handler_function, 88);
}

void notify() {
  if (!notification_pending.exchange(true)) {
    if (write(pipe_fds[1], "x", 1) == -1) {
      // Called from background threads, can't throw R error.
      REprintf("Failed to write to pipe for pending Python calls notifier\n");
    }
  }
}

void deinitialize() {
  if (input_handler) {
    removeInputHandler(&R_InputHandlers, input_handler);
    input_handler = nullptr;
  }

  if (pipe_fds[0] != -1) {
    close(pipe_fds[0]);
    pipe_fds[0] = -1;
  }

  if (pipe_fds[1] != -1) {
    close(pipe_fds[1]);
    pipe_fds[1] = -1;
  }
}

#endif // end unix

} // namespace pending_py_calls_notifier
