context("classes")

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
class Person:
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
class Clock:
  def __init__ (self, time):
    self.time = time
    
class Calendar:
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






