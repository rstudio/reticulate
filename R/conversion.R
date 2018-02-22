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
  py_ref_to_r(x)
}



#' @export
r_to_py.factor <- function(x, convert = FALSE) {
  r_to_py_impl(as.character(x), convert = convert)
}



#' @export
r_to_py.POSIXt <- function(x, convert = FALSE) {
  datetime <- import("datetime", convert = convert)
  datetime$datetime$fromtimestamp(as.double(x))
}

#' @export
py_to_r.datetime.datetime <- function(x) {
  time <- import("time", convert = TRUE)
  posix <- time$mktime(x$timetuple())
  posix <- posix + as.numeric(as_r_value(x$microsecond)) * 1E-6
  as.POSIXct(posix, origin = "1970-01-01")
}



#' @export
r_to_py.Date <- function(x, convert = FALSE) {
  datetime <- import("datetime", convert = convert)
  iso <- strsplit(format(x), "-", fixed = TRUE)[[1]]
  year <- as.integer(iso[[1]])
  month <- as.integer(iso[[2]])
  day <- as.integer(iso[[3]])
  datetime$date(year, month, day)
}

#' @export
py_to_r.datetime.date <- function(x) {
  iso <- py_to_r(x$isoformat())
  as.Date(iso)
}



#' @export
r_to_py.data.frame <- function(x, convert = FALSE) {
  
  # if we don't have pandas, just use default implementation
  if (!py_module_available("pandas"))
    return(r_to_py_impl(x, convert = convert))
  
  pd <- import("pandas", convert = convert)
  
  # manually convert each column to associated Python vector type
  columns <- lapply(x, function(column) {
    if (is.factor(column)) {
      pd$Categorical(as.character(column), categories = levels(column))
    } else if (is.numeric(column) || is.character(column)) {
      np_array(column)
    } else {
      r_to_py(column)
    }
  })
  
  # otherwise, attempt conversion using pandas APIs
  pdf <- pd$DataFrame$from_dict(columns)
  pdf$reindex(columns = names(x))
  
}

#' @export
py_to_r.pandas.core.frame.DataFrame <- function(x) {
  disable_conversion_scope(x)
  
  # extract numpy arrays associated with each column
  columns <- as_r_value(x$columns$values)
  items <- lapply(columns, function(column) {
    x[[column]]$as_matrix()
  })
  names(items) <- columns
  
  # convert back to R
  converted <- py_ref_to_r(dict(items))
  
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
      converted[[i]] <- factor(converted[[i]], levels = levels)
    }
    
  }
  
  # re-order based on ordering of pandas DataFrame
  as.data.frame(converted[columns], optional = TRUE, stringsAsFactors = FALSE)
}
