
#ifndef __PYTHON_EVENT_LOOP__
#define __PYTHON_EVENT_LOOP__

namespace event_loop {

void initialize();

struct Task {
  Task(void (*func)(void*), void* data) 
    : func(func), data(data)
  {
  }
  void (*func)(void*);
  void* data;
};

void register_task(Task task);

} // namespace event_loop

#endif // __PYTHON_EVENT_LOOP__
