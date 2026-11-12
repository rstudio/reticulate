
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
#' @section NumPy columns in tibbles:
#' On R >= 4.3, unconverted NumPy arrays can be columns in a
#' [tibble::tibble()]. The first dimension gives the number of rows; any
#' remaining dimensions belong to each row. For example:
#'
#' ```r
#' np <- import("numpy", convert = FALSE)
#' frames <- np$zeros(c(3L, 64L, 64L), dtype = "uint8")
#' videos <- tibble::tibble(frame = 1:3, pixels = frames)
#' dplyr::filter(videos, frame > 1L)
#' ```
#'
#' Tibble row subsetting and dplyr operations such as `filter()`, `arrange()`,
#' and `slice()` preserve the NumPy dtype and trailing dimensions. A lazy
#' vctrs proxy keeps array contents in Python, including dtypes without an
#' exact R equivalent. Row binding requires matching dtypes and trailing
#' dimensions. Use NumPy to cast or reshape arrays before binding if needed.
#'
#' NumPy columns retain Python reference semantics: changing the original
#' array also changes a column that refers to it. Subsetting rows creates a
#' new NumPy array. No automatic conversion takes place when constructing
#' a tibble; use `convert = FALSE` when importing NumPy.
#'
#' Use ordinary R columns for grouping, sorting, and join keys. Missing rows
#' and vctrs comparisons of NumPy columns are not supported. This includes
#' joins that introduce missing rows and `vctrs::vec_c()`, which initializes
#' missing rows internally; use `dplyr::bind_rows()` for complete columns.
#' Keep these columns in tibbles: base data-frame row subsetting uses Python's
#' indexing methods, which have different semantics. Direct `[` indexing on
#' a NumPy array continues to use Python's zero-based indexing.
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

#' @exportS3Method utils::str
str.numpy.ndarray <- function(object, ..., nest.lev = 0) {
  shape <- paste0(collapse = ",", as.integer(py_to_r(py_get_attr(object, "shape"))))
  dtype <- as.character(py_to_r(py_get_attr(py_get_attr(object, "dtype"), "name")))
  cat(sep = "",
      if (nest.lev > 0) " ",
      sprintf("<numpy.ndarray shape(%s), dtype=%s>\n", shape, dtype))
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
