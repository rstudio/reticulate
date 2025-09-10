context("vectors")

test_that("Single element vectors are treated as scalars", {
  skip_if_no_python()
  expect_true(test$isScalar(5))
  expect_true(test$isScalar(5L))
  expect_true(test$isScalar("5"))
  expect_true(test$isScalar(TRUE))
})

test_that("Multi-element vectors are treated as lists", {
  skip_if_no_python()
  expect_true(test$isList(c(5,5)))
  expect_true(test$isList(c(5L,5L)))
  expect_true(test$isList(c("5", "5")))
  expect_true(test$isList(c(TRUE, TRUE)))
})

test_that("The list function forces single-element vectors to be lists", {
  skip_if_no_python()
  expect_false(test$isScalar(list(5)))
  expect_false(test$isScalar(list(5L)))
  expect_false(test$isScalar(list("5")))
  expect_false(test$isScalar(list(TRUE)))
})

test_that("Large Python integers are converted to R numeric", {
  skip_if_no_python()
  
  # Test case from issue #1835
  py_run_string('def cum_square(n): return sum([x**2 for x in range(n + 1)])')
  
  # Small value that fits in R integer
  small_result <- py$cum_square(100L)
  expect_type(small_result, "integer")
  expect_equal(small_result, 338350L)
  
  # Large value that doesn't fit in R integer (would cause overflow)
  large_result <- py$cum_square(10000L)
  expect_type(large_result, "double")  # Should be converted to numeric
  expect_true(large_result > 0)  # Should be positive, not negative due to overflow
  expect_equal(large_result, 333383335000)  # Expected value
  
  # Test with explicit large integers
  py_run_string('large_int = 2**32')  # Larger than 32-bit signed int max
  large_int <- py$large_int
  expect_type(large_int, "double")
  expect_equal(large_int, 2^32)
  
  # Test edge cases around 32-bit integer limits
  py_run_string('max_r_int = 2147483647')  # 2^31 - 1, should fit in R integer
  max_r_int <- py$max_r_int  
  expect_type(max_r_int, "integer")
  expect_equal(max_r_int, 2147483647L)
  
  py_run_string('over_max_r_int = 2147483648')  # 2^31, should be double
  over_max_r_int <- py$over_max_r_int
  expect_type(over_max_r_int, "double")
  expect_equal(over_max_r_int, 2147483648)
  
  py_run_string('min_r_int = -2147483648')  # -2^31, should fit in R integer  
  min_r_int <- py$min_r_int
  expect_type(min_r_int, "integer")
  expect_equal(min_r_int, -2147483648L)
  
  py_run_string('under_min_r_int = -2147483649')  # -2^31 - 1, should be double
  under_min_r_int <- py$under_min_r_int
  expect_type(under_min_r_int, "double")
  expect_equal(under_min_r_int, -2147483649)
})

test_that("Lists with mixed size integers are converted to R numeric", {
  skip_if_no_python()
  
  # List with both small and large integers should become all numeric
  py_run_string('mixed_list = [100, 2147483648, 200]')  # mix of small and large ints
  mixed_result <- py$mixed_list
  expect_type(mixed_result, "double")
  expect_equal(mixed_result, c(100, 2147483648, 200))
})

