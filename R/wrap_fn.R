get_signature <- function(sigs) {
  signature <- ""
  sig_names <- names(sigs)
  for(k in sig_names) {
    if (isTRUE(sigs[[k]] == ""))
      # arg without default
      signature <- paste0(signature, k)
    else {
      # arg with default
      py_value_str <- ifelse(
        is.character(sigs[[k]]),
        paste0("'", sigs[[k]], "'"),
        as.character(r_to_py(eval(sigs[[k]])))) 
      signature <- paste0(signature, k, "=", py_value_str)
    }
    # if this is not the last arg, append a comma
    if (k != sig_names[length(sig_names)]) 
      signature <- paste0(signature, ", ")
  }
  signature
}

#' Wrap an R function in a Python function with the same signature.
#' 
#' This function could wrap an R function in a Python function with
#' the same signature. Note that if the signature of the R function
#' contain esoteric Python-incompatible constructs, the signature of 
#' the returned function will be `function(...)`.
#' 
#' @param f An R function
#' @return A Python function that calls the R function `f` with the same signature.
#' 
wrap_fn <- function(f) {
  tryCatch({
    sigs <- formals(f)
    if (is.null(sigs)) {
      func_signature <- func_pass_args <- ""
    } else {
      func_signature <- get_signature(sigs)
      func_pass_args <- get_signature(lapply(sigs, function(sig) ""))
    }
    wrap_fn_util <- py_run_string(sprintf("
def wrap_fn(f):
  def fn(%s):
    return f(%s)
  return fn
", func_signature, func_pass_args))
    wrap_fn_util$wrap_fn(f)
  }, error = function(e) r_to_py(f))
}
