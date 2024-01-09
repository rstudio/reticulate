test_that("py_to_r(<r_extptr_capsule>) returns the extptr", {

  # Mock class for testing
  Rcpp::sourceCpp(code='
#include <Rcpp.h>
using namespace Rcpp;

class AClass {
public:
  AClass() {
    Rprintf("AClass created");
  }
  ~AClass() {
    Rprintf("AClass destroyed");
  }
};

// [[Rcpp::export]]
SEXP getA() {
  AClass* ptr = new AClass;
  return Rcpp::XPtr< AClass >(ptr);
}
', env = environment())

  expect_output(x <- getA(), "AClass created")
  expect_output({
    xpy <- r_to_py(x)
    x_ <- py_to_r(xpy)
  }, NA)
  expect_reference(x, x_) # same memory address

  # test that gc()ing the py capsule doesn't gc() the extptr if there is
  # a live R reference to it
  expect_output({ rm(x, xpy); for(i in 1:3) gc(full = TRUE) }, NA)
  expect_output({ rm(x_)    ; for(i in 1:3) gc(full = TRUE) }, "AClass destroyed")


  # test that gc()ing the R ref to the extptr doesn't actuall gc() the extptr
  # object if the py capsule has a live reference to it.
  expect_output(x <- getA(), "AClass created")
  expect_output({
    xpy <- r_to_py(x)
    x_ <- py_to_r(xpy)
  }, NA)
  expect_reference(x, x_) # same memory address
  expect_output({ rm(x, x_); for(i in 1:3) gc(full = TRUE) }, NA)
  expect_output({ rm(xpy)  ; for(i in 1:3) gc(full = TRUE) }, "AClass destroyed")

})
