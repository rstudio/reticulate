
test_that("r helper accesses 'active' environment for tests", {
  rA <- 100
  pA <- py_eval("r.rA", convert = TRUE)
  expect_equal(rA, pA)
})
