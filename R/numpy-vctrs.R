# NumPy arrays are vectors of first-axis rows for vctrs. The proxy is a lazy
# list of row views, so even dtypes without an R equivalent stay in NumPy.
# Register these methods in .onLoad() because vctrs and pillar are suggested.
vec_proxy.numpy.ndarray <- function(x, ...) {
  shape <- dim(x)
  if (!length(shape))
    stop("A scalar NumPy array has no first dimension for dataframe rows.")
  numpy_array_proxy(x, shape[[1L]])
}

vec_restore.numpy.ndarray <- function(x, to, ...) {
  array <- numpy_array_proxy_data(x)
  if (!is.null(array))
    return(array)
  if (!length(x))
    return(py_get_item(to, py_slice(0L, 0L)))
  if (any(vapply(x, is.null, logical(1))))
    stop("NumPy dataframe columns do not support missing rows.")
  # concatenate() normalizes byte order; retain the original dtype as well.
  import("numpy", convert = FALSE)$concatenate(x, axis = 0L)$astype(
    py_get_attr(to, "dtype"), copy = FALSE
  )
}

numpy_array_proxy_row <- function(array, index) {
  py_get_item(array, py_slice(index - 1L, index))
}

numpy_array_proxy_slice <- function(array, indices) {
  indices[indices > dim(array)[[1L]]] <- NA_integer_
  # vctrs also slices proxies with missing indices to allocate row-binding
  # output. Keep those as NULL until restoration, after assignment is done.
  if (anyNA(indices))
    return(lapply(indices, function(i) {
      if (is.na(i)) NULL else numpy_array_proxy_row(array, i)
    }))
  rows <- import("numpy", convert = FALSE)$take(
    array, np_array(as.integer(indices) - 1L), axis = 0L
  )
  numpy_array_proxy(rows, length(indices))
}

vec_ptype2.numpy.ndarray.numpy.ndarray <- function(x, y, ...) {
  numpy_array_check_compatible(x, y)
  py_get_item(x, py_slice(0L, 0L))
}

vec_cast.numpy.ndarray.numpy.ndarray <- function(x, to, ...) {
  numpy_array_check_compatible(x, to)
  x
}

numpy_array_check_compatible <- function(x, y) {
  if (!identical(dim(x)[-1L], dim(y)[-1L]))
    stop("NumPy dataframe columns must have the same trailing dimensions.")
  if (!py_to_r(py_get_attr(x, "dtype") == py_get_attr(y, "dtype")))
    stop("NumPy dataframe columns must have the same dtype.")
}

# Comparing the row-view environments would compare object identities, not
# array values. Use ordinary metadata columns as keys instead.
vec_proxy_equal.numpy.ndarray <- function(x, ...) {
  stop("NumPy dataframe columns do not support equality comparisons in vctrs.")
}

vec_proxy_compare.numpy.ndarray <- function(x, ...) {
  stop("NumPy dataframe columns do not support ordering comparisons in vctrs.")
}

pillar_shaft.numpy.ndarray <- function(x, ...) {
  shape <- dim(x)
  dtype <- py_to_r(py_get_attr(py_get_attr(x, "dtype"), "name"))
  label <- if (length(shape) == 1L) {
    paste0("<", dtype, ">")
  } else {
    paste0("<", dtype, " [", paste(shape[-1L], collapse = "x"), "]>")
  }
  pillar::new_pillar_shaft_simple(rep(label, shape[[1L]]), align = "left")
}
