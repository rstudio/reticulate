

test_that("py_allow_threads() can enable/disable background threads", {

  file <- tempfile()
  on.exit(unlink(file), add = TRUE)

  write_to_file_from_thread <- py_run_string("
def write_to_file_from_thread(path, lines):
    from time import sleep, localtime, strftime

    def write_to_file(path, lines):
        sleep(.1) # don't try to run until we've had a chance to return to the R main thread
        with open(path, 'w') as f:
            for line in list(lines):
                f.write(line + '\\n')

    from _thread import start_new_thread
    start_new_thread(write_to_file, (path, lines))
", local = TRUE)$write_to_file_from_thread

  reticulate:::py_allow_threads_impl(FALSE)
  write_to_file_from_thread(file, letters)
  Sys.sleep(.3)
  # confirm background thread did not run while R was sleeping
  expect_false(file.exists(file))
  # explicitly enter python and release the gil
  import("time")$sleep(.1)
  # confirm the background thread ran on py_sleep()
  expect_identical(readLines(file), letters)

  unlink(file)

  reticulate:::py_allow_threads_impl(TRUE)
  write_to_file_from_thread(file, letters)
  Sys.sleep(.3)
  # confirm that the background thread ran while R was sleeping.
  expect_identical(readLines(file), letters)

})
