
#' Tensor shape
#'
#' @param ... Tensor dimensions
#'
#' @export
shape <- function(...) {
  values <- list(...)
  lapply(values, function(value) {
    if (!is.null(value))
      as.integer(value)
    else
      NULL
  })
}
