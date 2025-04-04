
#' NumPy array
#'
#' Create NumPy arrays and convert the data type and in-memory
#' ordering of existing NumPy arrays.
#'
#' @param data Vector or existing NumPy array providing data for the array
#' @param dtype Numpy data type (e.g. "float32", "float64", etc.)
#' @param order Memory ordering for array. "C" means C order, "F" means Fortran
#'   order.
#'
#' @return A NumPy array object.
#'
#' @export
np_array <- function(data, dtype = NULL, order = "C") {

  # convert to numpy if required
  if (!inherits(data, "numpy.ndarray")) {

    # check if this object has object bit set (skip dispatch
    # if we know it's unnecessary)
    isobj <- is.object(data)

    # convert non-array to array
    if (!is.array(data))
      data <- as.array(data)

    # do the conversion (will result in Fortran column ordering)
    data <- if (isobj) {
      r_to_py(data, convert = FALSE)
    } else {
      r_to_py_impl(data, convert = FALSE)
    }
  }

  # if we don't yet have a dtype then use the converted type
  if (is.null(dtype))
    dtype <- data$dtype

  # convert data as necessary (this will be a no-op and will
  # return the input array if dtype and order are already
  # satisfied)
  data$astype(dtype = dtype, order = order, copy = FALSE)
}


#' @export
length.numpy.ndarray <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    length(NULL)
  else
    as_r_value(x$size)
}


#' @export
dim.numpy.ndarray <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    return(NULL)
  as.integer(py_to_r(py_get_attr(x, "shape")))
}

#' @export
t.numpy.ndarray <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    return(NULL)
  py_get_attr(x, "T")
}


#' Reshape an Array
#'
#' Reshape (reindex) a multi-dimensional array, using row-major (C-style) reshaping
#' semantics by default.
#'
#' This function differs from e.g. `dim(x) <- dim` in a very important way: by
#' default, `array_reshape()` will fill the new dimensions in row-major (`C`-style)
#' ordering, while [dim<-()] will fill new dimensions in column-major
#' (`F`ortran-style) ordering. This is done to be consistent with libraries
#' like NumPy, Keras, and TensorFlow, which default to this sort of ordering when
#' reshaping arrays. See the examples for why this difference may be important.
#'
#' @param x An array
#' @param dim The new dimensions to be set on the array.
#' @param order The order in which elements of `x` should be read during
#'   the rearrangement. `"C"` means elements should be read in row-major
#'   order, with the last index changing fastest; `"F"` means elements should
#'   be read in column-major order, with the first index changing fastest.
#'
#' @examples \dontrun{
#' # let's construct a 2x2 array from a vector of 4 elements
#' x <- 1:4
#'
#' # rearrange will fill the array row-wise
#' array_reshape(x, c(2, 2))
#' #      [,1] [,2]
#' # [1,]    1    2
#' # [2,]    3    4
#' # setting the dimensions 'fills' the array col-wise
#' dim(x) <- c(2, 2)
#' x
#' #      [,1] [,2]
#' # [1,]    1    3
#' # [2,]    2    4
#' }
#' @export
array_reshape <- function(x, dim, order = c("C", "F")) {
  order <- match.arg(order)
  np <- import("numpy", convert = !is_py_object(x))
  np$reshape(x, as.integer(dim), order = order)
}
