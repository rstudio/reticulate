
# convenience function for importing tensorflow
#' @export
tensorflow <- function(module = NULL) {
  if (is.null(module))
    py_import("tensorflow")
  else
    py_import(paste("tensorflow", module, sep="."))
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
  tensorflow()$add(a, b)
}

#' @export
"+.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tensorflow()$add(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  if (missing(b))
    tensorflow()$neg(a)
  else
    tensorflow()$sub(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  if (missing(b))
    tensorflow()$neg(a)
  else
    tensorflow()$sub(a, b)
}

#' @export
"*.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tensorflow()$mul(a, b)
}

#' @export
"*.tensorflow.python.ops.variables.Variable" <- function(a, b) {
  tensorflow()$mul(a, b)
}

#' @export
"/.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tensorflow()$truediv(a, b)
}

#' @export
"%/%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tensorflow()$floordiv(a, b)
}

#' @export
"%%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tensorflow()$mod(a, b)
}

