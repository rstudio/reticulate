
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

  operator PyObject*() const { return pObject_; }

private:
  PyObject* pObject_;
  bool owned_;
};


// singleton
class PythonInterpreter;
PythonInterpreter& pythonInterpreter();

class PythonInterpreter : boost::noncopyable {

// singleton
private:
  PythonInterpreter();
  friend PythonInterpreter& pythonInterpreter();

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
  PythonObject mainModule_;
  PythonObject mainDictionary_;
};





#endif // __PYTHON_HPP__
