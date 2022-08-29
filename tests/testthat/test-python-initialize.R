context("py_initialize")

test_that("sys.executable points to the correct python", {


  py_exe_stand_alone <-  system2(py_exe(), c("-c", shQuote("import sys; print(sys.executable)")), stdout = TRUE)
  py_exe_embedded <- import("sys")$executable

  if(is_windows())
    py_exe_embedded <- utils::shortPathName(py_exe_embedded)

   expect_identical(py_exe_stand_alone, py_exe_embedded)

})

