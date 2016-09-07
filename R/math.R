
#' @export
tf.square <- function(x, name=NULL) {
  tf$square(x, name = name)
}

#' @export
tf.log <- function(x, name=NULL) {
  tf$log(x, name = name)
}


#' @export
tf.reduce_mean <- function(input_tensor, reduction_indices=NULL,
                           keep_dims=FALSE, name=NULL) {
  tf$reduce_mean(input_tensor,
                 reduction_indices = reduction_indices,
                 keep_dims = keep_dims,
                 name = name)
}

#' @export
tf.reduce_sum <- function(input_tensor, reduction_indices=NULL, keep_dims=FALSE,
                          name=NULL) {
  tf$reduce_sum(input_tensor,
                reduction_indices = reduction_indices,
                keep_dims = keep_dims,
                name = name)
}


#' @export
tf.matmul <- function(a, b,
                      transpose_a=FALSE, transpose_b=FALSE,
                      a_is_sparse=FALSE, b_is_sparse=FALSE,
                      name=NULL) {
  tf$matmul(a, b,
            transpose_a = transpose_a, transpose_b = transpose_b,
            a_is_sparse = a_is_sparse, b_is_sparse = b_is_sparse,
            name = name)
}

#' @export
tf.argmax <- function(input, dimension, name=NULL) {
  tf$argmax(input, dimension = dimension, name = name)
}


# Some math generics, see here for docs on more generics:
# https://stat.ethz.ch/R-manual/R-devel/library/base/html/groupGeneric.html

# __neg__ (unary -)
# __abs__ (abs())
# __invert__ (unary ~)
# __add__ (binary +)
# __sub__ (binary -)
# __mul__ (binary elementwise *)
# __div__ (binary / in Python 2)
# __floordiv__ (binary // in Python 3)
# __truediv__ (binary / in Python 3)
# __mod__ (binary %)
# __pow__ (binary **)
# __and__ (binary &)
# __or__ (binary |)
# __xor__ (binary ^)
# __lt__ (binary <)
# __le__ (binary <=)
# __gt__ (binary >)
# __ge__ (binary >=)

#' @export
"+.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$add(a, b)
}

#' @export
"+.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$add(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  if (missing(b))
    tf$neg(a)
  else
    tf$sub(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  if (missing(b))
    tf$neg(a)
  else
    tf$sub(a, b)
}

#' @export
"*.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$mul(a, b)
}

#' @export
"*.tensorflow.python.ops.variables.Variable" <- function(a, b) {
  tf$mul(a, b)
}

#' @export
"/.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$truediv(a, b)
}

#' @export
"%/%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$floordiv(a, b)
}

#' @export
"%%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$mod(a, b)
}
