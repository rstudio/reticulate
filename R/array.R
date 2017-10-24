
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
    
    # convert non-array to array
    if (!is.array(data))
      data <- as.array(data)
    
    # do the conversion (will result in Fortran column ordering)
    data <- r_to_py(data)
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
"length.numpy.ndarray" <- function(x) {
  if (py_is_null_xptr(x))
    length(NULL)
  else
    as_r_value(x$size)
}

#' Reshape an Array
#' 
#' Reshape (reindex) a multi-dimensional array, using reshaping semantics that
#' align with the behavior of NumPy's own `reshape()` function.
#' 
#' This function differs from e.g. `dim(x) <- dim` in a very important way: by
#' default, `rearray()` will fill the new dimensions in row-major
#' (`F`ortran-style) ordering, while [dim<-()] will fill new dimensions in
#' column-major (`C`-style) ordering. This is done to be consistent with the
#' NumPy `np.reshape()` function, which defaults to this sort of ordering when
#' reshaping arrays. See the examples for why this difference may be important.
#' 
#' @param x Either an \R array or a NumPy array.
#' @param dim The new dimensions to be set on the array.
#' @param order The order in which elements of `x` should be read during
#'   the rearrangement. `"C"` means elements should be read in row-major
#'   order, with the last index changing fastest; `"F"` means elements should
#'   be read in column-major order, with the first index changing fastest.
#' 
#' @examples
#' # let's construct a 2x2 array from a vector of 4 elements
#' x <- 1:4
#' 
#' # rearrange will fill the array row-wise
#' rearray(x, c(2, 2))
#' #      [,1] [,2]
#' # [1,]    1    2
#' # [2,]    3    4
#' # setting the dimensions 'fills' the array col-wise
#' dim(x) <- c(2, 2)
#' x
#' #      [,1] [,2]
#' # [1,]    1    3
#' # [2,]    2    4
#' @export
rearray <- function(x, dim, order = c("C", "F")) {
  np <- import("numpy", convert = FALSE)
  order <- match.arg(order)
  reshaped <- np$reshape(x, as.integer(dim), order)
  if (!inherits(x, "python.builtin.object"))
    reshaped <- py_to_r(reshaped)
  reshaped
}

