
#ifndef __PYTHON_HPP__
#define __PYTHON_HPP__

#include <Python.h>

#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

class PythonObject : boost::noncopyable {

public:
  explicit PythonObject(PyObject* pObject, bool owned = true)
    : pObject_(pObject), owned_(owned)
  {
  }

  virtual ~PythonObject();

  PyObject* get() const { return pObject_; }
  operator PyObject*() const { return get(); }

private:
  PyObject* pObject_;
  bool owned_;
};

class PythonModule : public PythonObject {
public:
  // attach to existing module (reference not owned)
  explicit PythonModule(PyObject* module);

  // import module
  explicit PythonModule(const char* name);

private:
  PythonObject dictionary_;
};


// singleton
class PythonInterpreter;
PythonInterpreter& pythonInterpreter();

class PythonInterpreter : boost::noncopyable {

// singleton
private:
  PythonInterpreter();
  friend PythonInterpreter& pythonInterpreter();

public:
  // code execution
  void execute(const std::string& code);
  void executeFile(const std::string& file);

  // get the main module
  boost::shared_ptr<PythonModule> mainModule() const { return pMainModule_; }

private:
  class PythonSession : boost::noncopyable {
  public:
    PythonSession() { ::Py_Initialize(); }
    ~PythonSession() { ::Py_Finalize(); }
  };
  PythonSession session_;
  boost::shared_ptr<PythonModule> pMainModule_;
};


#endif // __PYTHON_HPP__
