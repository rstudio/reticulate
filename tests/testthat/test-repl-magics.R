context("repl_python() magics")

quiet_repl <- function() {
  options("reticulate.repl.quiet" = TRUE)
  sink(nullfile())
}

if(getRversion() < "3.6")
nullfile <- function()
  if (.Platform$OS.type == "windows") "nul:" else "/dev/null"

unquiet_repl <- function() {
  options("reticulate.repl.quiet" = NULL)
  sink()
}

local_quiet_repl <- function(envir = parent.frame()) {
  quiet_repl()
  withr::defer(unquiet_repl(), envir = envir)
}


test_that("%pwd, %cd", {

  owd <- getwd()
  local_quiet_repl()


  expect_output(
    repl_python(input = "%pwd"),
    paste0(">>>  %pwd\n", owd),
    fixed = TRUE)

  expect_error(
    repl_python(input = "%pwd foo"), "no arguments")

  repl_python(input = c(
    "x = %pwd",
    "%cd ..",
    "y = %pwd",
    "%cd -",
    "z = %pwd"
  ))

  expect_equal(py_eval("x"), owd)
  expect_equal(py_eval("y"), dirname(owd))
  expect_equal(py_eval("z"), owd)

  setwd(owd)

})



test_that("%env", {

  local_quiet_repl()

  repl_python(input = c(
    "x = %env FOOVAR",
    "%env FOOVAR baz",
    "y = %env FOOVAR",
    "%env FOOVAR=foo",
    "z = %env FOOVAR"
    ))

  expect_equal(py_eval("x"), "")
  expect_equal(py_eval("y"), "baz")
  expect_equal(py_eval("z"), "foo")
  Sys.unsetenv("FOOVAR")

})

test_that("%system, !", {

  local_quiet_repl()

  repl_python(input = "x = !ls")
  expect_equal(py_eval("x"), system("ls", intern = TRUE))

})


test_that("%pip", {

  skip_if_no_test_environments()
  local_quiet_repl()

  env_path <- virtualenv_create("test-pip-repl-magic")

  expect_true(callr::r(function(env_path) {
    Sys.unsetenv("RETICULATE_PYTHON")
    library(reticulate)

    use_virtualenv(env_path, required = TRUE)

    repl_python(input = "%pip install requests")
    import("requests")
    TRUE
  }, args = list(env_path = env_path)))

  virtualenv_remove(env_path, confirm = FALSE)
  # unlink(env_path, recursive = TRUE)

})


test_that("%conda", {

  skip_if_no_test_environments()
  skip_if_no_conda()
  local_quiet_repl()

  capture.output({
    python <- conda_create("test-conda-repl-magic")
  })

  expect_true(callr::r(function(python) {
    Sys.unsetenv("RETICULATE_PYTHON")
    library(reticulate)

    use_condaenv(python, required = TRUE)

    # TODO: pass through interactive response from the user for prompts like:
    # Proceed ([y]/n)?
    repl_python(input = "%conda install -y rsa")
    import("rsa")
    TRUE
  },
  stdout = tempfile("conda output"),
  args = list(python = python)))

  capture.output({
    conda_remove("test-conda-repl-magic")
  })

  # info <- get("get_python_conda_info",asNamespace("reticulate"))(python)
  # unlink(info$root, recursive = TRUE)
})

test_that("!! respects string literals", {

  local_quiet_repl()

  repl_python(input = 'x = "!!"')
  expect_identical(py_eval("x"), "!!")

  repl_python(input = '_ = "ab!!cd!!ef"')

  expect_identical(py_eval("_"), "ab!!cd!!ef")

  files <- system("ls", intern = TRUE)
  repl_python(input = "files = !!ls")
  expect_equal(py_eval("files"), files)

  repl_python(input = "first_file, *other_files = !!ls")
  expect_equal(py_eval("first_file"), files[1])
  expect_equal(py_eval("other_files"), files[-1])

  repl_python(input = "(first_file, *other_files) = !!ls")
  expect_equal(py_eval("first_file"), files[1])
  expect_equal(py_eval("other_files"), files[-1])
})

test_that("repl_expand_bangbang handles assignment forms", {
  expect_identical(
    reticulate:::repl_expand_bangbang("obj.attr = !!cmd"),
    "obj.attr = %system cmd"
  )
  expect_identical(
    reticulate:::repl_expand_bangbang("first, second = !!cmd"),
    "first, second = %system cmd"
  )
  expect_identical(
    reticulate:::repl_expand_bangbang("single, = !!cmd"),
    "single, = %system cmd"
  )
  expect_identical(
    reticulate:::repl_expand_bangbang("  result = !!cmd"),
    "  result = %system cmd"
  )
  expect_identical(
    reticulate:::repl_expand_bangbang("value = \"!!\""),
    "value = \"!!\""
  )
})
