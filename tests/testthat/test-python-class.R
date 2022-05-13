context("PyClass")

test_that("Can create a Python class and call methods", {
  skip_if_no_python()
  
  Hi <- PyClass("Hi", list(
    name = NULL,
    `__init__` = function(self, name) {
      self$name <- name
      NULL
    },
    say_hi = function(self) {
      paste0("Hi ", self$name)
    }
  ))
  
  a <- Hi("World")
  b <- Hi("R")
  
  expect_equal(a$say_hi(), "Hi World")
  expect_equal(b$say_hi(), "Hi R")
  
})

test_that("Can inherit from another class created in R", {
  skip_if_no_python()
  
  Hi <- PyClass("Hi", list(
    name = NULL,
    `__init__` = function(self, name) {
      self$name <- name
      NULL
    },
    say_hi = function(self) {
      paste0("Hi ", self$name)
    }
  ))
  
  Hello <- PyClass("Hello", inherit = Hi, list(
    say_hello = function(self) {
      paste0("Hello and ", super()$say_hi())
    }
  ))
  
  a <- Hello("World")
  expect_equal(a$say_hello(), "Hello and Hi World")
})

test_that("Can inherit from a Python class", {
  skip_if_no_python()
  
  py <- reticulate::py_run_string("
class Person(object):
  def __init__ (self, name):
    self.name = name
")
  
  Person2 <- PyClass("Person2", inherit = py$Person, list(
    `__init__` = function(self, name) {
      super()$`__init__`(name)
      self$test <- name
      NULL
    }
  ))
  

  a <- Person2("World")
  expect_equal(a$name, "World")
  expect_equal(a$test, "World")
})

test_that("Can inherit from multiple Python classes", {
  skip_if_no_python()
  
  py <- reticulate::py_run_string("
class Clock(object):
  def __init__ (self, time):
    self.time = time
    
class Calendar(object):
  def __init__ (self, date):
    self.date = date
")
  
  ClockCalendar <- PyClass("ClockCalendar", inherit = list(py$Clock, py$Calendar), list(
    `__init__` = function(self, time, date) {
      py$Clock$`__init__`(self, time)
      py$Calendar$`__init__`(self, date)
    }
  ))
  
  a <- ClockCalendar("15:54", "2019-11-05")
  
  expect_equal(a$date, "2019-11-05")
  expect_equal(a$time, "15:54")
})

test_that("Can define and instantiate empty classes", {
  skip_if_no_python()
  
  Hi <- PyClass("Hi")
  x <- Hi()
  x$name <- "Hi"
  
  expect_s3_class(x, "python.builtin.Hi")
  expect_equal(x$name, "Hi")
})

test_that("Methods can access enclosed env", {
  skip_if_no_python()
  
  Hi <- PyClass("Hi", list(
    say_a = function(self) {
      a
    }
  ))
  
  x <- Hi()
  
  a <- 1
  expect_equal(x$say_a(), 1)
  a <- 2
  expect_equal(x$say_a(), 2)
})

test_that("Can directly access variables from a class", {
  
  skip_if_no_python()
  
  bt <- import_builtins()
  
  Hi <- PyClass("Hi", list(
    `__init__` = function(self, a) {
      self$a <- a
      self$x <- bt$isinstance # Have a python that can't be 
                              # converted to an R type
      NULL
    },
    add_2 = function(self) {
      self$a + 2
    },
    get_class = function(self) {
      class(self$a)
    },
    add_n = function(self, n) {
      self$a + n
    }
  ))
  
  x <- Hi(a = 2)
  expect_equal(x$add_2(), 4)
  expect_equal(x$a, 2)
  expect_equal(x$get_class(), "numeric")
  expect_equal(x$add_n(10), 12)
})

test_that("Properties are automatically converted in inherited classes", {
  skip_if_no_python()
  
  bt <- import_builtins(convert = FALSE)
  p <- py_run_string("
class Base(object):
  def __init__ (self, x):
    self.x = 1
", convert = FALSE)
  
  Hi <- PyClass("Hi", inherit = p$Base, list(
    `__init__` = function(self, a, ...) {
      self$a <- a
      self$k <- bt$isinstance # Have a python that can't be
                              # converted to an R type
      super()$`__init__`(...)
    },
    add_2 = function(self) {
      self$a + 2
    },
    get_class = function(self) {
      class(self$a)
    },
    add_n = function(self, n) {
      self$a + n
    }
  ))
  
  x <- Hi(a = 2, x = 1)
  expect_equal(x$add_2(), 4)
  expect_equal(x$a, 2)
  expect_equal(x$get_class(), "numeric")
  expect_equal(x$add_n(10), 12)
})

test_that("self is not converted when there's a py_to_r method for it", {
  skip_if_no_python()
  
  bt <- import_builtins(convert = FALSE)
  Base <- PyClass("Base")
  
  Hi <- PyClass("Hi", inherit = Base, list(
    `__init__` = function(self, a) {
      self$a <- a
      NULL
    },
    add_2 = function(self) {
      self$a + 2
    }
  ))
  
  assign("py_to_r.python.builtin.Base", value = function(x) {
    "base"
  }, envir = .GlobalEnv)
  
  x <- Hi(2)
  expect_equal(x, "base")
  
  rm(py_to_r.python.builtin.Base, envir = .GlobalEnv)
})

test_that("can call super from an inherited class", {
  
  Base1 <- PyClass("Base1", list(`__init__` = function(self, a) {
    self$a <- a
  }))
  
  Base2 <- PyClass("Base2", list(`__init__` = function(self, a, b) {
    self$b <- b
    super()$`__init__`(a)
  }), inherit = Base1)
  
  Inhe <- PyClass("Inhe", inherit = Base2, list(
    `__init__` = function(self, a, b, c) {
      self$c <- c
      super()$`__init__`(a, b)
      NULL
    }
  ))
  
  x <- Inhe(10, 20, 30)
  
  
  expect_equal(x$a, 10)
  expect_equal(x$b, 20)
  expect_equal(x$c, 30)
})



