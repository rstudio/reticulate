
#ifndef __PYTHON_HPP__
#define __PYTHON_HPP__

#include <Python.h>

#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

class PythonObject : boost::noncopyable {

public:
  explicit PythonObject(PyObject* pObject, bool borrowed = false)
    : pObject_(pObject), borrowed_(borrowed)
  {
  }

  virtual ~PythonObject();

  PyObject* get() const { return pObject_; }

private:
  PyObject* pObject_;
  bool borrowed_;
};

class PythonModule : public PythonObject {
public:
  explicit PythonModule(const char* name);

private:
  PythonObject dictionary_;
};

// singleton
class PythonInterpreter;
PythonInterpreter& pythonInterpreter();

class PythonInterpreter : boost::noncopyable {

// construction/destruction
public:
  PythonInterpreter();

// singleton
private:
  friend PythonInterpreter& python();

  // public interface
public:
  void execute(const std::string& code);
  void executeFile(const std::string& file);

private:
  class PythonSession : boost::noncopyable {
  public:
    PythonSession() { ::Py_Initialize(); }
    ~PythonSession() { ::Py_Finalize(); }
  };
  PythonSession session_;
  PythonModule mainModule_;
};





#endif // __PYTHON_HPP__
