get_signature <- function(sigs) {
  signature <- ""
  sig_names <- names(sigs)
  for(k in sig_names) {
    if (sigs[[k]] == "") 
      # arg without default
      signature <- paste0(signature, k)
    else {
      # arg with default
      signature <- paste0(signature, k, "=", as.character(r_to_py(sigs[[k]])))
    }
    # if this is not the last arg, append a comma
    if (k != sig_names[length(sig_names)]) 
      signature <- paste0(signature, ", ")
  }
  signature
}

#' Wrap R function into Python function with the same signature.
#' 
#' This function could wrap an R function into Python function with
#' the same signature. Note that this is function still experimental.
#' 
#' @param f An R function
#' @return A Python function with the same signature as `f`.
#' 
wrap_fn <- function(f) {
  sigs <- formals(f)
  wrap_fn_util <- py_run_string(sprintf("
def wrap_fn(f):
  def fn(%s):
    return f(%s)
  return fn
", get_signature(sigs), get_signature(lapply(sigs, function(sig) ""))))
  wrap_fn_util$wrap_fn(f)
}