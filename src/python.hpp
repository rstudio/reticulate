
#ifndef __PYTHON_HPP__
#define __PYTHON_HPP__

#include <boost/noncopyable.hpp>

// singleton
class Python;
Python& python();

// dynamically loaded interface to python shared library
class Python : boost::noncopyable {

// construction/destruction
public:
  Python();
  virtual ~Python();

// non-copyable singleton
private:
  friend Python& python();
  Python(const Python& other);      // non construction-copyable
  Python& operator=(const Python&); // non copyable
};

#endif // __PYTHON_HPP__
