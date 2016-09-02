
#' @export
square <- function(x, name=NULL) {
  tf <- tf_import()
  tf$square(x, name = name)
}

#' @export
reduce_mean <- function(input_tensor, reduction_indices=NULL,
                        keep_dims=FALSE, name=NULL) {
  tf <- tf_import()
  tf$reduce_mean(input_tensor,
                 reduction_indices = as_integer(reduction_indices),
                 keep_dims = keep_dims,
                 name = name)
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
  tf <- tf_import()
  tf$add(a, b)
}

#' @export
"+.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf <- tf_import()
  tf$add(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  if (missing(b))
    tf$neg(a)
  else
    tf$sub(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf <- tf_import()
  if (missing(b))
    tf$neg(a)
  else
    tf$sub(a, b)
}

#' @export
"*.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$mul(a, b)
}

#' @export
"*.tensorflow.python.ops.variables.Variable" <- function(a, b) {
  tf <- tf_import()
  tf$mul(a, b)
}

#' @export
"/.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$truediv(a, b)
}

#' @export
"%/%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$floordiv(a, b)
}

#' @export
"%%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$mod(a, b)
}
