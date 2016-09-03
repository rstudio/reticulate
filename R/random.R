
#' @export
random_uniform <- function(shape, minval=0, maxval=NULL, dtype="float32",
                           seed=NULL, name=NULL) {
  dtype <- match.arg(dtype, c("float32", "float64", "int32", "int64"))
  tf$random_uniform(as_shape(shape), minval = minval, maxval = maxval,
                    dtype = as_dtype(dtype), seed = seed, name = name)
}

