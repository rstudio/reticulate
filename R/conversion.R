#' Convert between Python and R objects
#'
#' @inheritParams import
#' @param x A Python object.
#'
#' @return An \R object, as converted from the Python object.
#'
#' @name r-py-conversion
#' @export
r_to_py <- function(x, convert = FALSE) {
  ensure_python_initialized()
  UseMethod("r_to_py")
}

#' @rdname r-py-conversion
#' @export
py_to_r <- function(x) {
  if(!is_py_object(x <- py_to_r_cpp(x))) # simple obj
    return(x)
  # Note, the new `x` here (with convert=TRUE) won't be visible to S3 methods
  # UseMethod() doesn't allow modifying the original call args
  UseMethod("py_to_r")
}


#' @export
r_to_py.default <- function(x, convert = FALSE) {
  r_to_py_impl(x, convert = convert)
}

#' @export
py_to_r.default <- function(x) {
  py_to_r_cpp(x)
}

#' R wrapper for Python objects
#'
#' S3 method to create a custom R wrapper for a Python object.
#' The default wrapper is either an R environment or an R function
#' (for callable python objects).
#'
#' @param x Python object
#'
#' @keywords internal
#'
#' @export
py_to_r_wrapper <- function(x) {
  UseMethod("py_to_r_wrapper")
}

#' @export
py_to_r_wrapper.default <- function(x) {
  x
}


#' @export
r_to_py.factor <- function(x, convert = FALSE) {
  if (inherits(x, "ordered"))
    warning("converting ordered factor to character; ordering will be lost")
  r_to_py_impl(as.character(x), convert = convert)
}



#' @export
r_to_py.POSIXt <- function(x, convert = FALSE) {
  tz <- attr(x, "tzone", TRUE)

  ## POSIXlt tzone is a length-3 vec, where "" means local/missing tzone
  if (length(tz))
    tz <- tz[[1L]] %""% NULL

  # we prefer numpy datetime64 for efficiency
  # numpy doesn't support tzone
  # (internally, always stored from UTC epoch, like POSIXct in R)
  if (is.null(tz) && py_module_available("numpy")) {
    np_array <- r_to_py_impl(array(as.double(x) * 1E9), convert = FALSE)
    return(np_array$astype(dtype = "datetime64[ns]"))
  }

  if (!is.null(tz)) {
    if (py_version() >= "3.9")
      tz <- import("zoneinfo", convert = FALSE)$ZoneInfo(tz)
    else
      tz <- import("pytz")$timezone(tz)
  }
  datetime <- import("datetime", convert = FALSE)
  x <- as.double(x)
  if (length(x) == 1L) {
    datetime$datetime$fromtimestamp(x, tz = tz)
  } else {
    r_to_py_impl(
      lapply(x, datetime$datetime$fromtimestamp, tz = tz),
      convert = FALSE
    )
  }
}


#' @export
py_to_r.datetime.datetime <- function(x) {
  if (py_version() >= "3") {
    tz <- NULL
    if (!is.null(x$tzinfo)) {

      # in Python 3.9, there is a new zoneinfo.ZoneInfo class that
      # accepts Olsonnames, similar to R's tz= semantics.
      # Try to find the user supplied value in that case.
      # Note that accessing `ZoneInfo.tzname()` is lossy. Eg.
      # doing `ZoneInfo("America/New_York").tzname()` returns "EDT", which is
      # not in R's OlsonNames() database, and also not stable wrt DST status.

      # pytz.zone, round-trips better
      if(is.null(tz))
        tz <- as_r_value(py_get_attr(x$tzinfo, "zone", TRUE))

      # zoneinfo.ZoneInfo(), roundtrip lossy (standardized), but in py stdlib
      if (is.null(tz))
        tz <- as_r_value(py_get_attr(x$tzinfo, "key", TRUE))

      # common to pytz.timezone, ZoneInfo, other subclasses
      if (is.null(tz))
        tryCatch(tz <- as_r_value(x$tzname()), error = identity)
    }

    # TODO: if tzname() raised NotImplemented,
    #   - restore last user facing python exception,
    #   - have a fallback trying to construct a tz string w/ utcoffset().
    return(.POSIXct(as_r_value(x$timestamp()), tz = tz))
  }

  # old python2 compat code.
  # mangles tzinfo attribute: https://github.com/rstudio/reticulate/issues/1265
  disable_conversion_scope(x)

  # convert to POSIX time
  time <- import("time", convert = TRUE)
  posix <- time$mktime(x$timetuple())

  # include microseconds as well
  ms <- as_r_value(x$microsecond)
  posix <- posix + as.numeric(ms) * 1E-6

  # TODO: handle non-UTC timezones?
  as.POSIXct(posix, origin = "1970-01-01", tz = "UTC")

}



#' @export
r_to_py.Date <- function(x, convert = FALSE) {
  r_convert_date(x, convert)
}

#' @export
py_to_r.datetime.date <- function(x) {
  iso <- as_r_value(x$isoformat())
  as.Date.character(iso)
}


#' @export
py_to_r.collections.OrderedDict <- function(x) {
  keys <- py_dict_get_keys(x)
  py_set_convert(keys, FALSE)

  local_conversion_scope(x, TRUE)
  result <- lapply(seq_len(length(keys)) - 1L, function(i) {
    py_dict_get_item(x, keys[i])
  })

  names(result) <- py_dict_get_keys_as_str(x)
  result
}


#' @export
py_to_r.pandas.core.series.Series <- function(x) {
  local_conversion_scope(x, FALSE)
  values <- py_to_r(x$values)
  index <- py_to_r(x$index)
  bt <- import("builtins")
  names(values) <- as.character(bt$list(index$map(bt$str)))
  values
}

#' @export
py_to_r.pandas.Series <- py_to_r.pandas.core.series.Series

#' @export
py_to_r.pandas.core.categorical.Categorical <- function(x) {
  local_conversion_scope(x, FALSE)
  values <- py_to_r(x$get_values())
  levels <- py_to_r(x$categories$values)
  ordered <- py_to_r(x$dtype$ordered)
  factor(values, levels = levels, ordered = ordered)
}

#' @export
py_to_r.pandas.core.arrays.categorical.Categorical <-
  py_to_r.pandas.core.categorical.Categorical

#' @export
py_to_r.pandas.arrays.NumpyExtensionArray <- function(x) {
  py_to_r(x$to_numpy())
}

#' @export
py_to_r.pandas._libs.missing.NAType <- function(x) {
  NA
}

#' @export
py_to_r.pandas._libs.missing.C_NAType <- function(x) {
  NA
}

py_object_shape <- function(object) {
  unlist(as_r_value(object$shape))
}

#' @export
summary.pandas.core.series.Series <- function(object, ...) {
  if (py_is_null_xptr(object) || !py_available())
    str(object)
  else
    object$describe()
}

#' @export
summary.pandas.Series <- summary.pandas.core.series.Series

#' @export
length.pandas.core.series.Series <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    0L
  else {
    py_object_shape(x)[[1]]
  }
}

#' @export
length.pandas.Series <- length.pandas.core.series.Series

#' @export
dim.pandas.core.series.Series <- function(x) {
  NULL
}

#' @export
dim.pandas.Series <- dim.pandas.core.series.Series

#' @export
r_to_py.data.frame <- function(x, convert = FALSE) {

  # if we don't have pandas, just use default implementation
  if (!py_module_available("pandas"))
    return(r_to_py_impl(x, convert = convert))

  pd <- import("pandas", convert = FALSE)

  # manually convert each column to associated Python vector type
  columns <- r_convert_dataframe(x, convert = convert)

  # generate DataFrame from dictionary
  pdf <- pd$DataFrame$from_dict(columns)

  # copy over row names if they exist
  rni <- .row_names_info(x, type = 0L)
  if (is.character(rni)) {
    if (length(rni) == 1)
      rni <- as.list(rni)
    pdf$index <- rni
  }

  # re-order based on original columns
  if (length(x) > 1)
    pdf <- pdf$reindex(columns = names(x))

  pdf

}

#' @export
py_to_r.datatable.Frame <- function(x) {
  # TODO: it would be nice to avoid the extra conversion to pandas
  py_to_r(x$to_pandas())
}

#' @export
py_to_r.pandas.core.frame.DataFrame <- function(x) {
  local_conversion_scope(x, FALSE)

  np <- import("numpy", convert = TRUE)
  pandas <- import("pandas", convert = TRUE)
  bt <- import("builtins", convert = TRUE)

  # delegate to c++
  converted <- py_convert_pandas_df(x)
  converted <- lapply(converted, function(col) {
    if (identical(length(dim(col)), 1L))
      dim(col) <- NULL
    col
  })

  # assign colnamnes
  columns <- x$columns$values
  columns <- as.character(bt$list(x$columns$map(bt$str)))
  names(converted) <- columns

  df <- converted
  attr(df, "row.names") <- c(NA_integer_, py_to_r(-x$shape[[0L]]))
  class(df) <- "data.frame"

  # attempt to copy over index, and set as rownames when appropriate
  #
  # TODO: should we tag the R data.frame with the original Python index
  # object in case users need it?
  #
  # TODO: Pandas allows for a large variety of index formats; we should
  # try to explicitly whitelist a small family which we can represent
  # effectively in R
  index <- x$index

  # tag the returned object with the Python index, in case
  # the user needs to explicitly access / munge the index
  # for some need
  attr(df, "pandas.index") <- index

  if (inherits(index, c("pandas.core.indexes.base.Index",
                        "pandas.indexes.base.Index",
                        "pandas.Index"
                      ))) {

    if (inherits(index, c("pandas.core.indexes.range.RangeIndex",
                          "pandas.indexes.range.RangeIndex",
                          "pandas.RangeIndex"
                        )) &&
        np$issubdtype(index$dtype, np$number))
    {
      # check for a range index from 0 -> n. in such a case, we don't need
      # to copy or translate the index. note that we need to translate from
      # Python's 0-based indexing to R's one-based indexing.
      #
      # NOTE: `_start` and friends were deprecated with Pandas 0.25.0,
      # with non-private versions preferred for access instead
      if (reticulate::py_has_attr(index, "start"))
      {
        start <- py_to_r(index[["start"]])
        stop  <- py_to_r(index[["stop"]])
        step  <- py_to_r(index[["step"]])
      }
      else
      {
        start <- py_to_r(index[["_start"]])
        stop  <- py_to_r(index[["_stop"]])
        step  <- py_to_r(index[["_step"]])
      }

      if (start != 0 || stop != nrow(df) || step != 1) {
        values <- tryCatch(py_to_r(index$values), error = identity)
        if (is.numeric(values)) {
          rownames(df) <- values + 1
        }
      }
    }

    else if (inherits(index, c("pandas.core.indexes.datetimes.DatetimeIndex",
                               "pandas.tseries.index.DatetimeIndex",
                               "pandas.DatetimeIndex"
                              ))) {

      converted <- tryCatch(py_to_r(index$values), error = identity)

      tz <- index[["tz"]]
      if (inherits(tz, "pytz.tzinfo.BaseTzInfo") ||
          inherits(tz, "pytz.UTC"))
      {
        zone <- tryCatch(py_to_r(tz$zone), error = function(e) NULL)
        if (!is.null(zone) && zone %in% OlsonNames())
          attr(converted, "tzone") <- zone
      }

      rownames(df) <- converted
    }

    else {
      converted <- tryCatch(py_to_r(index$values), error = identity)
      if (is.character(converted) || is.numeric(converted)) {
        if (any(duplicated(converted))) {
          warning("index contains duplicated values: row names not set")
        } else {
          rownames(df) <- converted
        }
      }

    }
  }

  df

}

#' @export
py_to_r.pandas.DataFrame <- py_to_r.pandas.core.frame.DataFrame

#' @export
summary.pandas.core.frame.DataFrame <- summary.pandas.core.series.Series

#' @export
summary.pandas.DataFrame <- summary.pandas.core.frame.DataFrame

#' @export
length.pandas.core.frame.DataFrame <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    0L
  else {
    py_object_shape(x)[[2]]
  }
}

#' @export
length.pandas.DataFrame <- length.pandas.core.frame.DataFrame

#' @export
dim.pandas.core.frame.DataFrame <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    NULL
  else
    py_object_shape(x)
}

#' @export
dim.pandas.DataFrame <- dim.pandas.core.frame.DataFrame


# Scipy sparse matrices
#' @importFrom Matrix Matrix

#' @export
dim.scipy.sparse.base.spmatrix <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    NULL
  else
    py_object_shape(x)
}

#' @export
length.scipy.sparse.base.spmatrix <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    0L
  else
    prod(py_object_shape(x))
}

#' @export
py_to_r.scipy.sparse.base.spmatrix <- function(x) {
  py_to_r(x$tocsc())
}

#' @importFrom methods as
#' @export
r_to_py.sparseMatrix <- function(x, convert = FALSE) {
  x <- if (package_version(as.vector(getNamespaceVersion("Matrix"))) >= "1.4-2")
    as(as(as(x, "dMatrix"), "generalMatrix"), "CsparseMatrix")
  else
    as(x, "dgCMatrix")
  r_to_py(x, convert = convert)
}

# Conversion between `Matrix::dgCMatrix` and `scipy.sparse.csc.csc_matrix`.
# Scipy CSC Matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html

#' @export
r_to_py.dgCMatrix <- function(x, convert = FALSE) {
  # use default implementation if scipy is not available
  if (!py_module_available("scipy"))
    return(r_to_py_impl(x, convert = convert))
  sp <- import("scipy.sparse", convert = FALSE)
  csc_x <- sp$csc_matrix(
    tuple(
      np_array(x@x), # Data array of the matrix
      np_array(x@i), # CSC format index array
      np_array(x@p), # CSC format index pointer array
      convert = FALSE
    ),
    shape = dim(x)
  )
  if (any(dim(x) != dim(csc_x)))
    stop(
      paste(
        "Failed to convert: dimensions of the original Matrix::dgCMatrix ",
        "object (", dim(x), ") and the converted Scipy CSC matrix (",
        dim(csc_x), ") do not match", sep="", collapse=", "))
  csc_x
}

#' @importFrom methods new
#' @export
py_to_r.scipy.sparse.csc.csc_matrix <- function(x) {
  disable_conversion_scope(x)

  x <- x$sorted_indices()
  new(
    "dgCMatrix",
    i = as.integer(as_r_value(x$indices)),
    p = as.integer(as_r_value(x$indptr)),
    x = as.vector(as_r_value(x$data)),
    Dim = as.integer(dim(x))
  )

}

# Conversion between `Matrix::dgRMatrix` and `scipy.sparse.csr.csr_matrix`.
# Scipy CSR Matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

#' @export
r_to_py.dgRMatrix <- function(x, convert = FALSE) {
  # use default implementation if scipy is not available
  if (!py_module_available("scipy"))
    return(r_to_py_impl(x, convert = convert))
  sp <- import("scipy.sparse", convert = FALSE)
  csr_x <- sp$csr_matrix(
    tuple(
      np_array(x@x), # Data array of the matrix
      np_array(x@j), # CSR format index array
      np_array(x@p), # CSR format index pointer array
      convert = FALSE
    ),
    shape = dim(x)
  )
  if (any(dim(x) != dim(csr_x)))
    stop(
      paste(
        "Failed to convert: dimensions of the original Matrix::dgRMatrix ",
        "object (", dim(x), ") and the converted Scipy CSR matrix (",
        dim(csr_x), ") do not match", sep="", collapse=", "))
  csr_x
}

#' @export
py_to_r.scipy.sparse.csr.csr_matrix <- function(x) {
  disable_conversion_scope(x)

  x <- x$sorted_indices()
  new(
    "dgRMatrix",
    j = as.integer(as_r_value(x$indices)),
    p = as.integer(as_r_value(x$indptr)),
    x = as.vector(as_r_value(x$data)),
    Dim = as.integer(dim(x))
  )

}

# Conversion between `Matrix::dgTMatrix` and `scipy.sparse.coo.coo_matrix`.
# Scipy COO Matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

#' @export
r_to_py.dgTMatrix <- function(x, convert = FALSE) {
  # use default implementation if scipy is not available
  if (!py_module_available("scipy"))
    return(r_to_py_impl(x, convert = convert))
  sp <- import("scipy.sparse", convert = FALSE)
  coo_x <- sp$coo_matrix(
    tuple(
      np_array(x@x), # Data array of the matrix
      tuple(
        np_array(x@i),
        np_array(x@j),
        convert = FALSE
      ),
      convert = FALSE
    ),
    shape = dim(x)
  )
  if (any(dim(x) != dim(coo_x)))
    stop(
      paste(
        "Failed to convert: dimensions of the original Matrix::dgTMatrix ",
        "object (", dim(x), ") and the converted Scipy COO matrix (",
        dim(coo_x), ") do not match", sep="", collapse=", "))
  coo_x
}

#' @importFrom methods new
#' @export
py_to_r.scipy.sparse.coo.coo_matrix <- function(x) {
  disable_conversion_scope(x)

  new(
    "dgTMatrix",
    i = as.integer(as_r_value(x$row)),
    j = as.integer(as_r_value(x$col)),
    x = as.vector(as_r_value(x$data)),
    Dim = as.integer(dim(x))
  )

}



r_convert_dataframe_column <- function(column, convert) {

  pd <- import("pandas", convert = FALSE)
  if (is.factor(column)) {
    pd$Categorical(as.list(as.character(column)),
                   categories = as.list(levels(column)),
                   ordered = inherits(column, "ordered"))
  } else if (is.numeric(column) || is.character(column)) {
    np_array(column)
  } else if (inherits(column, "POSIXt")) {
    np_array(as.numeric(column) * 1E9, dtype = "datetime64[ns]")
  } else {
    r_to_py(column)
  }

}

# workaround for deprecation of packages in scipy 1.8.0
#' @export
dim.scipy.sparse._base.spmatrix <- dim.scipy.sparse.base.spmatrix
#' @export
length.scipy.sparse._base.spmatrix <- length.scipy.sparse.base.spmatrix
#' @export
py_to_r.scipy.sparse._base.spmatrix <- py_to_r.scipy.sparse.base.spmatrix
#' @export
py_to_r.scipy.sparse._csc.csc_matrix <- py_to_r.scipy.sparse.csc.csc_matrix
#' @export
py_to_r.scipy.sparse._csr.csr_matrix <- py_to_r.scipy.sparse.csr.csr_matrix
#' @export
py_to_r.scipy.sparse._coo.coo_matrix <- py_to_r.scipy.sparse.coo.coo_matrix

# updated locations for scipy 1.11.0
#' @export
dim.scipy.sparse._base._spbase <- dim.scipy.sparse.base.spmatrix
#' @export
length.scipy.sparse._base._spbase <- length.scipy.sparse.base.spmatrix
#' @export
py_to_r.scipy.sparse._matrix.spmatrix <- py_to_r.scipy.sparse.base.spmatrix

#' @export
as.data.frame.polars.dataframe.frame.DataFrame <-
function(x, row.names = NULL, optional = FALSE, ...) {
  py_set_convert(x$to_pandas, TRUE)()
}
