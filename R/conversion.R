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
  ensure_python_initialized()
  UseMethod("py_to_r")
}



#' @export
r_to_py.default <- function(x, convert = FALSE) {
  r_to_py_impl(x, convert = convert)
}

#' @export
py_to_r.default <- function(x) {
  if (!inherits(x, "python.builtin.object"))
    stop("Object to convert is not a Python object")
  
  # get the default wrapper
  x <- py_ref_to_r(x)
    
  # allow customization of the wrapper
  wrapper <- py_to_r_wrapper(x)
  attributes(wrapper) <- attributes(x)
  
  # return the wrapper
  wrapper 
}

#' R wrapper for Python objects 
#' 
#' S3 method to create a custom R wrapper for a Python object.
#' The default wrapper is either an R environment or an R function
#' (for callable python objects).
#' 
#' @param x Python object 
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
py_to_r.numpy.ndarray <- function(x) {
  disable_conversion_scope(x)
  
  # handle numpy datetime64 objects. fortunately, as per the
  # numpy documentation:
  #
  #    Datetimes are always stored based on POSIX time
  #
  # although some work is required to handle the different
  # subtypes of datetime64 (since the units since epoch can
  # be configurable)
  #
  # TODO: Python (by default) displays times using UTC time;
  # to reflect that behavior we also us 'tz = "UTC"', but we
  # might consider just using the default (local timezone)
  np <- import("numpy", convert = TRUE)
  if (np$issubdtype(x$dtype, np$datetime64)) {
    vector <- py_to_r(x$astype("datetime64[ns]")$astype("float64"))
    return(as.POSIXct(vector / 1E9, origin = "1970-01-01", tz = "UTC"))
  }
  
  # no special handler found; delegate to next method
  NextMethod()
}



#' @export
r_to_py.POSIXt <- function(x, convert = FALSE) {
  
  # we prefer datetime64 for efficiency
  if (py_module_available("numpy"))
    return(np_array(as.numeric(x) * 1E9, dtype = "datetime64[ns]"))
  
  datetime <- import("datetime", convert = convert)
  datetime$datetime$fromtimestamp(as.double(x))
}

#' @export
py_to_r.datetime.datetime <- function(x) {
  disable_conversion_scope(x)
  time <- import("time", convert = TRUE)
  posix <- time$mktime(x$timetuple())
  posix <- posix + as.numeric(as_r_value(x$microsecond)) * 1E-6
  as.POSIXct(posix, origin = "1970-01-01")
}



#' @export
r_to_py.Date <- function(x, convert = FALSE) {
  
  # we prefer datetime64 for efficiency
  if (py_module_available("numpy"))
    return(r_to_py.POSIXt(as.POSIXct(x)))
  
  # otherwise, fallback to using Python's datetime class
  datetime <- import("datetime", convert = convert)
  items <- lapply(x, function(item) {
    iso <- strsplit(format(x), "-", fixed = TRUE)[[1]]
    year <- as.integer(iso[[1]])
    month <- as.integer(iso[[2]])
    day <- as.integer(iso[[3]])
    datetime$date(year, month, day)
  })
  
  if (length(items) == 1)
    items[[1]]
  else
    items
}

#' @export
py_to_r.datetime.date <- function(x) {
  disable_conversion_scope(x)
  iso <- py_to_r(x$isoformat())
  as.Date(iso)
}



#' @export
py_to_r.pandas.core.series.Series <- function(x) {
  disable_conversion_scope(x)
  py_to_r(x$as_matrix())
}



#' @export
r_to_py.data.frame <- function(x, convert = FALSE) {
  
  # if we don't have pandas, just use default implementation
  if (!py_module_available("pandas"))
    return(r_to_py_impl(x, convert = convert))
  
  pd <- import("pandas", convert = FALSE)
  
  # manually convert each column to associated Python vector type
  columns <- lapply(x, function(column) {
    if (is.factor(column)) {
      pd$Categorical(as.character(column),
                     categories = as.list(levels(column)),
                     ordered = inherits(column, "ordered"))
    } else if (is.numeric(column) || is.character(column)) {
      np_array(column)
    } else if (inherits(column, "POSIXt")) {
      np_array(as.numeric(column) * 1E9, dtype = "datetime64[ns]")
    } else {
      r_to_py(column)
    }
  })
  
  # generate DataFrame from dictionary
  pdf <- pd$DataFrame$from_dict(columns)
  
  # copy over row names if they exist
  rni <- .row_names_info(x, type = 0L)
  if (is.character(rni))
    pdf$index <- rni
  
  # re-order based on original columns
  if (length(x) > 1)
    pdf <- pdf$reindex(columns = names(x))
  
  pdf
  
}

#' @export
py_to_r.pandas.core.frame.DataFrame <- function(x) {
  disable_conversion_scope(x)
  
  np <- import("numpy", convert = TRUE)
  
  # extract numpy arrays associated with each column
  columns <- py_to_r(x$columns$values)
  converted <- lapply(columns, function(column) {
    py_to_r(x[[column]]$as_matrix())
  })
  names(converted) <- columns
  
  # clean up converted objects
  for (i in seq_along(converted)) {
    column <- names(converted)[[i]]
    
    # drop 1D dimensions
    if (identical(dim(converted[[i]]), length(converted[[i]]))) {
      dim(converted[[i]]) <- NULL
    }
    
    # convert categorical variables to factors
    if (identical(py_to_r(x[[column]]$dtype$name), "category")) {
      levels <- py_to_r(x[[column]]$values$categories$values)
      ordered <- py_to_r(x[[column]]$dtype$ordered)
      converted[[i]] <- factor(converted[[i]], levels = levels, ordered = ordered)
    }
    
  }
  
  # re-order based on ordering of pandas DataFrame
  df <- as.data.frame(
    converted[columns],
    optional = TRUE,
    stringsAsFactors = FALSE
  )
  
  # attempt to copy over index, and set as rownames when appropriate
  #
  # TODO: should we tag the R data.frame with the original Python index
  # object in case users need it?
  #
  # TODO: Pandas allows for a large variety of index formats; we should
  # try to explicitly whitelist a small family which we can represent
  # effectively in R
  index <- x$index
  if (inherits(index, "pandas.core.indexes.base.Index")) {
    
    if (inherits(index, "pandas.core.indexes.range.RangeIndex") &&
        np$issubdtype(index$dtype, np$number))
    {
      # check for a range index from 0 -> n. in such a case, we don't need
      # to copy or translate the index. note that we need to translate from
      # Python's 0-based indexing to R's one-based indexing
      start <- py_to_r(index[["_start"]])
      stop  <- py_to_r(index[["_stop"]])
      step  <- py_to_r(index[["_step"]])
      if (start != 0 || stop != nrow(df) || step != 1) {
        values <- tryCatch(py_to_r(index$values), error = identity)
        if (is.numeric(values)) {
          rownames(df) <- values + 1
        }
      }
    }
    
    else if (inherits(index, "pandas.core.indexes.datetimes.DatetimeIndex")) {
      py_tz <- index[["tz"]]
      tz <- NULL
      
      if(inherits(py_tz, "pytz.tzinfo.BaseTzInfo") || 
         inherits(py_tz, "pytz.UTC")) {
        tz <- tryCatch(py_to_r(py_tz$zone), error = identity)
      }
      
      converted <- tryCatch(py_to_r(index$values), error = identity)
      
      if(!is.null(tz) && tz %in% OlsonNames()) {
        attr(converted, "tzone") <- tz
      }
      rownames(df) <- converted
    }
    
    else {
      converted <- tryCatch(py_to_r(index$values), error = identity)
      if (is.character(converted) || is.numeric(converted))
        rownames(df) <- converted
    }
  }
  
  df
  
}
