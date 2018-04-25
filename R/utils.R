

traceback_enabled <- function() {

  # if there is specific option set then respect it
  reticulate_traceback_option <- getOption("reticulate.traceback", default = NULL)
  if (!is.null(reticulate_traceback_option))
    return(isTRUE(reticulate_traceback_option))

  # determine whether rstudio python traceback support is available
  # and whether rstudio tracebacks are currently enabled
  rstudio_has_python_tracebacks <- exists(".rs.getActivePythonStackTrace",
                                          mode = "function")
  if (rstudio_has_python_tracebacks) {

    error_option_code <- deparse(getOption("error"))
    error_option_code_has <- function(pattern) {
      any(grepl(pattern, error_option_code))
    }
    rstudio_traceback_enabled <- error_option_code_has("\\.rs\\.recordTraceback")

    # if it is then we disable tracebacks
    if (rstudio_traceback_enabled)
      return(FALSE)
  }

  # default to tracebacks enabled
  TRUE
}

clear_error_handler <- function(retvalue = NA) {
  function(e) {
    py_clear_last_error()
    if (!is.null(retvalue) && is.na(retvalue))
      e
    else
      retvalue
  }
}

as_r_value <- function(x) {
  if (inherits(x, "python.builtin.object"))
    py_to_r(x)
  else
    x
}

yoink <- function(package, symbol) {
  do.call(":::", list(package, symbol))
}

defer <- function(expr, envir = parent.frame()) {
  call <- substitute(
    evalq(expr, envir = envir),
    list(expr = substitute(expr), envir = parent.frame())
  )
  do.call(base::on.exit, list(substitute(call), add = TRUE), envir = envir)
}

#' @importFrom utils head
disable_conversion_scope <- function(object) {

  if (!inherits(object, "python.builtin.object"))
    return(FALSE)

  envir <- as.environment(object)
  if (exists("convert", envir = envir, inherits = FALSE)) {
    convert <- get("convert", envir = envir)
    assign("convert", FALSE, envir = envir)
    defer(assign("convert", convert, envir = envir), envir = parent.frame())
  }

  TRUE
}

new_stack <- function() {

  (function() {

    .data <- list()

    methods <- list(
      clear  = function() { .data <<- character() },
      data   = function() { .data },
      empty  = function() { length(.data) == 0 },
      length = function() { length(.data) },
      push   = function(line) { .data[[length(.data) + 1]] <<- line },
      peek   = function() { .data[[length(.data)]] },
      pop    = function() { .data <<- utils::head(.data, n = -1) },
      set    = function(data) { .data <<- data }
    )

    list2env(methods)

  })()

}

