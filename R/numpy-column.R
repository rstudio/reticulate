#' NumPy arrays as dataframe columns
#'
#' Wrap a NumPy array as an R vector of rows for use in data frames and
#' tibbles. Requires R >= 4.3 and vctrs >= 0.7.0.
#'
#' @param x A plain NumPy ndarray with at least one dimension. Import NumPy
#'   with `convert = FALSE` to obtain unconverted arrays. NumPy subclasses,
#'   such as masked arrays, must be converted explicitly before wrapping.
#'
#' @return An `np_array_column` vector, with one element per first-axis row.
#'
#' @details
#' The wrapper uses R's one-based row indexing. Its length is the number of
#' rows, and it has no dimensions or names. The dtype and trailing dimensions
#' belong to each row. Bare NumPy arrays retain their usual Python semantics.
#'
#' Slicing, printing, binding, and grouping manipulate shared array references
#' and row indices without copying array contents or creating a Python object
#' per row. Metadata can use memory proportional to the number of rows.
#' Row binding requires matching dtypes and trailing dimensions.
#'
#' Use [r_to_py()] to retrieve the selected rows as a NumPy array. An unchanged
#' column can return its original array; other selections are gathered in
#' batches. [np_array()] also accepts a column and applies its `dtype` and
#' `order` arguments. `as.array()` converts the selected values to an R array.
#' Passing a column to a Python function performs the same conversion as
#' `r_to_py()`.
#'
#' Arithmetic with another column or an R scalar runs in NumPy and returns a
#' column. Columns must have matching trailing dimensions and compatible row
#' counts (equal lengths, or one column with a single row). This also works in
#' grouped dplyr mutations. Use `np_array()` when you need other NumPy operations
#' or explicit broadcasting.
#'
#' Missing rows, including those introduced by joins, are represented in the
#' wrapper and can be inspected with `is.na()`. Remove or replace missing rows
#' before conversion to NumPy, for example with `column[!is.na(column)]`.
#' NumPy has no missing-row representation for every dtype. Use ordinary
#' metadata columns for grouping, ordering, and equality comparisons;
#' comparing array columns in vctrs is not supported.
#'
#' Columns retain references to their source arrays, including after row
#' subsetting. Changes to those arrays are visible in the columns. Do not
#' resize a source array or change its dtype while it is wrapped. As with
#' other Python references, columns cannot be restored with `readRDS()`;
#' convert values to R or save the arrays with NumPy for persistent storage.
#'
#' @examples
#' \dontrun{
#' np <- import("numpy", convert = FALSE)
#' frames <- np$zeros(c(3L, 64L, 64L), dtype = "uint8")
#' videos <- tibble::tibble(frame = 1:3, pixels = np_array_column(frames))
#' selected <- videos[c(3L, 1L), ]
#' np_array(selected$pixels)
#' dplyr::mutate(dplyr::group_by(videos, frame), pixels = pixels + 1L)
#' }
#' @export
np_array_column <- function(x) {
  if (!identical(class(x)[[1L]], "numpy.ndarray"))
    stop("`x` must be a plain NumPy ndarray, not a subclass.")
  shape <- dim(x)
  if (!length(shape))
    stop("A scalar NumPy array has no first dimension for dataframe rows.")
  source <- numpy_column_sources(x, shape[[1L]])
  # Older vctrs versions can modify record fields in place during vec_assign().
  loadNamespace("vctrs", versionCheck = list(op = ">=", version = "0.7.0"))
  new_np_array_column(source, seq_len(shape[[1L]]),
                      py_get_attr(x, "dtype"), shape[-1L])
}

new_np_array_column <- function(source, index, dtype, shape) {
  vctrs::new_rcrd(list(source = source, index = index), dtype = dtype,
                  shape = shape, class = "np_array_column")
}

# Register optional-package methods in .onLoad(). Inherit R indexing and the
# source/index dataframe proxy from vctrs_rcrd; never proxy bare ndarrays.
vec_ptype.np_array_column <- function(x, ...) {
  new_np_array_column(list(), integer(), attr(x, "dtype"), attr(x, "shape"))
}

vec_ptype2.np_array_column.np_array_column <- function(x, y, ...) {
  numpy_column_check_compatible(x, y)
  vec_ptype.np_array_column(x)
}

vec_cast.np_array_column.np_array_column <- function(x, to, ...) {
  numpy_column_check_compatible(x, to)
  x
}

numpy_column_check_compatible <- function(x, y) {
  if (!identical(attr(x, "shape"), attr(y, "shape")))
    stop("NumPy columns must have the same trailing dimensions.")
  if (!py_to_r(attr(x, "dtype") == attr(y, "dtype")))
    stop("NumPy columns must have the same dtype.")
}

# Source identities and indices are storage metadata, not array values.
vec_proxy_equal.np_array_column <- function(x, ...) {
  stop("NumPy columns do not support equality comparisons in vctrs.")
}

vec_proxy_compare.np_array_column <- function(x, ...) {
  stop("NumPy columns do not support ordering comparisons in vctrs.")
}

#' @export
is.na.np_array_column <- function(x) {
  is.na(vctrs::field(x, "index"))
}

#' @export
anyNA.np_array_column <- function(x, recursive = FALSE) {
  anyNA(vctrs::field(x, "index"))
}

#' @export
r_to_py.np_array_column <- function(x, convert = FALSE) {
  index <- vctrs::field(x, "index")
  if (anyNA(index))
    stop("Cannot convert a NumPy column with missing rows to an ndarray.")
  source <- vctrs::field(x, "source")
  np <- import("numpy", convert = FALSE)
  shape <- c(length(index), attr(x, "shape"))
  dtype <- attr(x, "dtype")
  if (!length(index))
    return(r_to_py_impl(np$empty(shape, dtype = dtype), convert))

  # The common single-source case stays compact, even for very long columns.
  array <- numpy_column_source(source)
  if (!is.null(array))
    return(r_to_py_impl(numpy_column_gather(array, index), convert))

  # Group shared references by identity and gather once per source. Integer
  # advanced indexing reads strided inputs without copying the entire source.
  groups <- vctrs::vec_group_loc(source)
  if (nrow(groups) == 1L)
    return(r_to_py_impl(numpy_column_gather(groups$key[[1L]], index), convert))
  out <- np$empty(shape, dtype = dtype)
  for (i in seq_len(nrow(groups))) {
    loc <- groups$loc[[i]]
    rows <- numpy_column_gather(groups$key[[i]], index[loc])
    py_set_item(out, np_array(loc - 1L), rows)
  }
  r_to_py_impl(out, convert)
}

numpy_column_gather <- function(array, index) {
  if (identical(index, seq_len(dim(array)[[1L]])))
    array
  else
    py_get_item(array, np_array(index - 1L))
}

#' @export
as.array.np_array_column <- function(x, ...) {
  py_to_r(r_to_py(x))
}

#' @export
Ops.np_array_column <- function(e1, e2) {
  if (.Generic %in% c("==", "!=", "<", "<=", ">", ">="))
    stop("Use `np_array()` to compare NumPy column values.")
  if (!missing(e2) && inherits(e1, "np_array_column") &&
      inherits(e2, "np_array_column")) {
    if (!identical(attr(e1, "shape"), attr(e2, "shape")))
      stop("NumPy column arithmetic requires matching trailing dimensions.")
    vctrs::vec_size_common(e1, e2)
  }
  operand <- function(x) {
    if (inherits(x, "np_array_column"))
      return(r_to_py(x))
    stopifnot(is.atomic(x), length(x) == 1L)
    x
  }
  args <- if (missing(e2)) list(operand(e1)) else list(operand(e1), operand(e2))
  np_array_column(do.call(.Generic, args))
}

#' @export
format.np_array_column <- function(x, ...) {
  shape <- attr(x, "shape")
  dtype <- py_to_r(py_get_attr(attr(x, "dtype"), "name"))
  label <- if (!length(shape)) {
    paste0("<", dtype, ">")
  } else {
    paste0("<", dtype, " [", paste(shape, collapse = "x"), "]>")
  }
  out <- rep(label, length(x))
  out[is.na(x)] <- NA_character_
  out
}

type_sum.np_array_column <- function(x) "numpy"

pillar_shaft.np_array_column <- function(x, ...) {
  pillar::new_pillar_shaft_simple(format(x), align = "left")
}
