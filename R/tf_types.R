
#' Tensor shape
#'
#' @param ... Tensor dimensions
#'
#' @export
shape <- function(...) {
  dims <- list(...)
  lapply(dims, function(dim) {
    if (!is.null(dim))
      as.integer(dim)
    else
      NULL
  })
}
