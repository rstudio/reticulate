
#' @export
tf.random_uniform <- function(shape, minval=0, maxval=NULL, dtype=tf.float32,
                              seed=NULL, name=NULL) {
  tf$random_uniform(as_shape(shape), minval = minval, maxval = maxval,
                    dtype = dtype, seed = seed, name = name)
}

