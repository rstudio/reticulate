
#' NumPy array
#'
#' Create NumPy arrays and convert the data type, dimensions, and in-memory
#' ordering of existing NumPy arrays.
#'
#' @param data Vector or existing NumPy array providing data for the array
#' @param dim Integer vector with array dimensions.
#' @param dtype Numpy data type (e.g. "float32", "float64", etc.)
#' @param order Memory ordering for array. "C" means C order, "F" means Fortran
#'   order.
#'
#' @return A NumPy array object.
#'
#' @export
np_array <- function(data, dim = dim(data), dtype = NULL, order = "C") {
  
  # convert to numpy if required
  if (!inherits(data, "numpy.ndarray")) {
    
    # convert non-array to array
    if (!is.array(data))
      data <- as.array(data)
    
    # do the conversion (will result in Fortran column ordering)
    data <- r_to_py(data)
  }
  
  # if we don't yet have a dtype then use the converted type
  if (is.null(dtype))
    dtype <- data$dtype
  
  # reshape if necessary
  np <- import("numpy", convert = FALSE)
  if (!missing(dim))
    data <- np$reshape(data, as.integer(dim), order = order)
  
  # convert data as necessary (this will be a no-op and will
  # return the input array if dtype and order are already
  # satisfied)
  data$astype(dtype = dtype, order = order, copy = FALSE)
}

#' @export
"dim.numpy.ndarray" <- function(x) {
  if (py_is_null_xptr(x))
    NULL
  else {
    ndim <- as_r_value(x$ndim)
    if (ndim == 0)
      NULL
    else
      as.integer(as_r_value(x$shape))
  }
}

#' @export
"dim<-.numpy.ndarray" <- function(x, value) {
  if (!py_is_null_xptr(x))
    r_to_py(x$reshape(as.integer(value)))
  else
    x
}

#' @export
"length.numpy.ndarray" <- function(x) {
  if (py_is_null_xptr(x))
    length(NULL)
  else
    as_r_value(x$size)
}

