get_signature <- function(sigs) {
  sig_names <- names(sigs)
  signature_strings <- lapply(sig_names, function(k) {
    if (identical(sigs[[k]], quote(expr = )))
      # arg without default
      k
    else {
      # arg with default
      py_value_str <- ifelse(
        is.character(sigs[[k]]),
        paste0("'", sigs[[k]], "'"),
        as.character(r_to_py(eval(sigs[[k]]))))
      paste0(k, "=", py_value_str)
    }
  })
  paste(signature_strings, collapse = ", ")
}

#' Wrap an R function in a Python function with the same signature.
#'
#' This function could wrap an R function in a Python function with
#' the same signature. Note that the signature of the R function
#' must not contain esoteric Python-incompatible constructs.
#'
#' @param f An R function
#' @return A Python function that calls the R function `f` with the same signature.
#' @export
py_func <- function(f) {
  tryCatch({
    sigs <- formals(f)
    if (is.null(sigs)) {
      func_signature <- func_pass_args <- ""
    } else {
      func_signature <- get_signature(sigs)
      func_pass_args <- get_signature(
        lapply(sigs, function(sig) quote(expr =)))
    }
    wrap_fn_util <- py_run_string(sprintf("
def wrap_fn(f):
  def fn(%s):
    return f(%s)
  return fn
", func_signature, func_pass_args))
    wrap_fn_util$wrap_fn(f)
  }, error = function(e) {
    stop(paste0("The R function's signature must not contains esoteric ",
                "Python-incompatible constructs. Detailed traceback: \n",
                e$message))
  })
}

