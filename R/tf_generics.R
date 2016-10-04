
#' @export
str.tensorflow.builtin.module <- function(object, ...) {
  py_xptr_str(object,
              cat("Module(", py_str(py_get_attr(object, "__name__")),
                  ")\n", sep="")
  )
}

#' @export
str.tensorflow.python.framework.ops.Tensor <- function(object, ...) {
  py_xptr_str(object, cat(py_str(object), "\n", sep=""))
}

#' @export
str.tensorflow.python.ops.variables.Variable <- function(object, ...) {
  py_xptr_str(object,
              cat("Variable(shape=", py_str(object$get_shape()), ", ",
                  "dtype=", object$dtype$name, ")\n", sep = "")
  )
}

#' @export
"print.tensorflow.python.framework.ops.Tensor" <- function(x, ...) {
  str(x, ...)
  if (!is.null(tf$get_default_session())) {
    value <- tryCatch(x$eval(), error = function(e) NULL)
    if (!is.null(value))
      cat(" ", str(value), "\n", sep = "")
  }
}

#' @export
print.tensorflow.python.ops.variables.Variable <- print.tensorflow.python.framework.ops.Tensor

#' @export
.DollarNames.tensorflow.python.platform.flags._FlagValues <- function(x, pattern = "") {

  # skip if this is a NULL xptr
  if (py_is_null_xptr(x))
    return(character())

  # get the underlying flags and return the names
  flags <- x$`__flags`
  names(flags)
}

# https://stat.ethz.ch/R-manual/R-devel/library/base/html/InternalMethods.html


#' @export
"[.tensorflow.python.framework.ops.Tensor" <- function(x, i, j, ..., drop = TRUE) {

  # tensor shape as a vector
  x_size <- x$get_shape()$as_list()
  n_indices <- length(x_size)

  # Capture all indices beyond i and j (skip function, `x`, `drop`, `i` & `j`
  # from the arguments). This enables users to skip indices to get their defaults
  cl <- match.call()
  args <- as.list(cl)[-1]
  extra_indices <- args[!names(args) %in% c('x', 'i', 'j', 'drop')]

  # if i wasn't specified, make it NA (keep all values)
  if (missing(i)) i <- list(NA)
  else i <- list(i)

  # if j wasn't specified, but is required, keep all elements
  # if it isn't required, skip it
  if (missing(j)) {
    if (n_indices > 1) j <- list(NA)
    else j <- list()
  } else {
    j <- list(j)
  }

  # add these to i
  indices <- c(i, j, extra_indices)

  # evaluate any calls and replace any skipped indices (names) with NAs
  indices <- lapply(indices,
                    function (x) {
                      if(is.name(x)) NA
                      else if (is.call(x)) eval(x)
                      else x
                    })

  # pad out if too few indices
  n_indices_specified <- length(indices)
  if (n_indices_specified < n_indices) {
    missing <- seq_len(n_indices - n_indices_specified) + n_indices_specified
    indices[missing] <- NA
  }

  # error if wrong number of indices
  if (n_indices_specified !=  n_indices) {
    stop ('incorrect number of dimensions')
  }

  # check all the indices are numeric and finite (or NA)
  bad_indices <- vapply(indices,
                        function (x) !((is.numeric(x) && is.finite(x)) |
                                         (length(x) == 1 && is.na(x))),
                        TRUE)

  if (any(bad_indices)) {
    if (sum(bad_indices) == 1) {
      msg <- sprintf('index %i is not numeric and finite',
                     which(bad_indices))
    } else {
      msg <- sprintf('indices %s are not numeric',
                     paste(which(bad_indices), collapse = ', '))
    }
    stop (msg)
  }

  # strip out any names
  names(indices) = NULL


  # find index starting element on each dimension
  begin <- vapply(indices,
                  function (x) {
                    if (length(x) == 1 && is.na(x)) 0
                    else x[1]
                  },
                  0)

  # find slice end in each dimension
  end <- vapply(indices,
                function (x) {
                  if (length(x) == 1 && is.na(x)) Inf
                  else x[length(x)]
                },
                0)

  # truncate missing indices to be finite
  end <- pmin(end, x_size)

  # ensure the index is positive and increasing
  negative_indices <- end < 0 | begin < 0 #| end < begin
  decreasing_indices <- end < begin

  if (any(negative_indices)) {
    stop ('negative indexing of Tensors is not curently supported')
  }

  if (any(decreasing_indices)) {
    stop ('decreasing indexing of Tensors is not curently supported')
  }

  # add one to the ends to account for Python's exclusive upper bound
  end <- end + 1

  # convert to shapes
  begin_shape <- do.call('shape', as.list(begin))
  end_shape <- do.call('shape', as.list(end))

  # add stride length (always 1) so that the output is consistent with python API
  stride_shape <- as.list(rep(1L, n_indices))

  # get shrink mask as an integer representing a bitstring
  # if drop=TRUE, drop all *indices* specified as integers,
  # i.e. for a 2x3 Tensor x:
  #   x[1:1, ,drop=TRUE] => shape 1x3
  #   x[1, ,drop=TRUE] => shape 3
  if (drop) {
    # create bit mask as a logical vector, then collapse to an integer
    shrink <- vapply(indices,
                     function (x) {
                       length(x) == 1 && !is.na(x)
                     },
                     FALSE)
    shrink_integer <- sum(2 ^ (seq_along(shrink) - 1)[shrink])
  } else {
    shrink_integer <- 0
  }

  # return the slice
  tf$strided_slice(input_ = x,
                   begin = begin_shape,
                   end = end_shape,
                   strides = stride_shape,
                   shrink_axis_mask = shrink_integer)
}

# https://stat.ethz.ch/R-manual/R-devel/library/base/html/groupGeneric.html

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
"/.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$truediv(a, b)
}

#' @export
"%/%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$floordiv(a, b)
}

#' @export
"%/%.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$floordiv(a, b)
}

#' @export
"%%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$mod(a, b)
}

#' @export
"%%.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$mod(a, b)
}

#' @export
"^.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$pow(a, b)
}

#' @export
"^.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$pow(a, b)
}

#' @export
"&.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$logical_and(a, b)
}

#' @export
"&.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$logical_and(a, b)
}

#' @export
"|.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$logical_or(a, b)
}

#' @export
"!.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$logical_not(x)
}

#' @export
"!.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$logical_not(x)
}

#' @export
"|.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$logical_or(a, b)
}

#' @export
"|.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$logical_or(a, b)
}

#' @export
"|.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$logical_or(a, b)
}

#' @export
"|.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$logical_or(a, b)
}

#' @export
"==.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$equal(a, b)
}

#' @export
"==.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$equal(a, b)
}

#' @export
"!=.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$not_equal(a, b)
}

#' @export
"!=.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$not_equal(a, b)
}

#' @export
"<.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$less(a, b)
}

#' @export
"<.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$less(a, b)
}

#' @export
"<=.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$less_equal(a, b)
}

#' @export
"<=.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$less_equal(a, b)
}

#' @export
">.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$greater(a, b)
}

#' @export
">.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$greater(a, b)
}

#' @export
">=.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$greater_equal(a, b)
}

#' @export
">=.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$greater_equal(a, b)
}

#' @export
"abs.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$abs(x)
}

#' @export
"abs.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$abs(x)
}

#' @export
"sign.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$sign(x)
}

#' @export
"sign.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$sign(x)
}

#' @export
"sqrt.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$sqrt(x)
}

#' @export
"sqrt.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$sqrt(x)
}

#' @export
"floor.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$floor(x)
}

#' @export
"floor.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$floor(x)
}

#' @export
"ceiling.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$ceil(x)
}

#' @export
"ceiling.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$ceil(x)
}

#' @export
"round.tensorflow.python.framework.ops.Tensor" <- function(x, digits = 0) {
  if (digits != 0)
    stop("TensorFlow round only supports rounding to integers")
  tf$round(x)
}

#' @export
"round.tensorflow.python.framework.ops.Variable" <- function(x, digits = 0) {
  if (digits != 0)
    stop("TensorFlow round only supports rounding to integers")
  tf$round(x)
}

#' @export
"exp.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$exp(x)
}

#' @export
"exp.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$exp(x)
}

#' @export
"log.tensorflow.python.framework.ops.Tensor" <- function(x, base = exp(1)) {
  if (base != exp(1))
    stop("TensorFlow log suppports only natural logarithms")
  tf$log(x)
}

#' @export
"log.tensorflow.python.framework.ops.Variable" <- function(x, base = exp(1)) {
  if (base != exp(1))
    stop("TensorFlow log suppports only natural logarithms")
  tf$log(x)
}

#' @export
"cos.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$cos(x)
}

#' @export
"cos.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$cos(x)
}

#' @export
"sin.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$sin(x)
}

#' @export
"sin.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$sin(x)
}

#' @export
"tan.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$tan(x)
}

#' @export
"tan.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$tan(x)
}

#' @export
"acos.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$acos(x)
}

#' @export
"acos.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$acos(x)
}

#' @export
"asin.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$asin(x)
}

#' @export
"asin.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$asin(x)
}

#' @export
"atan.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$atan(x)
}

#' @export
"atan.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$atan(x)
}

#' @export
"lgamma.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$lgamma(x)
}

#' @export
"lgamma.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$lgamma(x)
}

#' @export
"digamma.tensorflow.python.framework.ops.Tensor" <- function(x) {
  tf$digamma(x)
}

#' @export
"digamma.tensorflow.python.framework.ops.Variable" <- function(x) {
  tf$digamma(x)
}
