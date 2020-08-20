
#' Import a Python module
#'
#' Import the specified Python module, making it available for use from \R.
#'
#' @param module The name of the Python module.
#' 
#' @param as An alias for module name (affects names of R classes). Note that
#'   this is an advanced parameter that should generally only be used in package
#'   development (since it affects the S3 name of the imported class and can
#'   therefore interfere with S3 method dispatching).
#'  
#' @param path The path from which the module should be imported.
#' 
#' @param convert Boolean; should Python objects be automatically converted
#'   to their \R equivalent? If set to `FALSE`, you can still manually convert
#'   Python objects to \R via the [py_to_r()] function.
#'   
#' @param delay_load Boolean; delay loading the module until it is first used?
#'   When `FALSE`, the module will be loaded immediately. See **Delay Load**
#'   for advanced usages.
#'
#'
#' @section Python Built-ins:
#' 
#' Python's built-in functions (e.g. `len()`) can be accessed via Python's
#' built-in module. Because the name of this module has changed between Python 2
#' and Python 3, we provide the function `import_builtins()` to abstract over
#' that name change.
#' 
#'
#' @section Delay Load:
#' 
#' The `delay_load` parameter accepts a variety of inputs. If you just need to
#' ensure your module is lazy-loaded (e.g. because you are a package author and
#' want to avoid initializing Python before the user has explicitly requested it),
#' then passing `TRUE` is normally the right choice.
#' 
#' You can also provide a list of named functions, which act as callbacks to be
#' run when the module is later loaded. For example:
#' 
#' ```
#' delay_load = list(
#'   
#'   # run before the module is loaded
#'   before_load = function() { ... }
#'   
#'   # run immediately after the module is loaded
#'   on_load = function() { ... }
#'   
#'   # run if an error occurs during module import
#'   on_error = function(error) { ... }
#'   
#' )
#' ```
#' 
#' Alternatively, if you supply only a single function, that will be treated as
#' an `on_load` handler.
#' 
#' 
#' @section Import from Path:
#' 
#' `import_from_path()` can be used in you need to import a module from an arbitrary
#' filesystem path. This is most commonly used when importing modules bundled with an
#' \R package -- for example:
#' 
#' ```
#' path <- system.file("python", package = <package>)
#' reticulate::import_from_path(<module>, path = path, delay_load = TRUE)
#' ```
#'
#' @return An \R object wrapping a Python module. Module attributes can be accessed
#'   via the `$` operator, or via [py_get_attr()].
#'
#' @examples
#' \dontrun{
#' main <- import_main()
#' sys <- import("sys")
#' }
#'
#' @export
import <- function(module, as = NULL, convert = TRUE, delay_load = FALSE) {
  
  # if there is an as argument then register a filter for it
  if (!is.null(as)) {
    register_class_filter(function(classes) {
      sub(paste0("^", module), as, classes)
    })
  }
  
  # resolve delay load
  delay_load_environment <- NULL
  delay_load_priority <- 0
  delay_load_functions <- NULL
  if (is.function(delay_load)) {
    delay_load_functions <- list(on_load = delay_load)
    delay_load <- TRUE
  } else if (is.list(delay_load)) {
    delay_load_environment <- delay_load$environment
    delay_load_functions <- delay_load
    if (!is.null(delay_load$priority))
      delay_load_priority <- delay_load$priority
    delay_load <- TRUE
  }
  
  # normal case (load immediately)
  if (!delay_load || is_python_initialized()) {
    
    # ensure that python is initialized (pass top level module as
    # a hint as to which version of python to choose)
    ensure_python_initialized(required_module = module)
    
    # import the module
    py_module_import(module, convert = convert)

  }
  
  # delay load case (wait until first access)
  else {
    if (is.null(.globals$delay_load_module) || (delay_load_priority > .globals$delay_load_priority)) {
      .globals$delay_load_module <- module
      .globals$delay_load_environment <- delay_load_environment
      .globals$delay_load_priority <- delay_load_priority
    }
    module_proxy <- new.env(parent = emptyenv())
    module_proxy$module <- module
    module_proxy$convert <- convert
    if (!is.null(delay_load_functions)) {
      module_proxy$get_module <- delay_load_functions$get_module
      module_proxy$before_load <- delay_load_functions$before_load
      module_proxy$on_load <- delay_load_functions$on_load
      module_proxy$on_error <- delay_load_functions$on_error
    }
    attr(module_proxy, "class") <- c("python.builtin.module", "python.builtin.object")
    module_proxy
  }
}


#' @rdname import
#' @export
import_main <- function(convert = TRUE) {
  ensure_python_initialized()
  import("__main__", convert = convert)
}

#' @rdname import
#' @export
import_builtins <- function(convert = TRUE) {
  ensure_python_initialized()
  if (is_python3())
    import("builtins", convert = convert)
  else
    import("__builtin__", convert = convert)
}


#' @rdname import
#' @export
import_from_path <- function(module,
                             path = ".",
                             convert = TRUE,
                             delay_load = FALSE)
{
  # handle delayed import specially
  delay <-
    !identical(delay_load, FALSE) &&
    !py_available(initialize = FALSE)
  
  if (delay)
    import_from_path_delayed(module, path, convert, delay_load)
  else
    import_from_path_immediate(module, path, convert)
}

import_from_path_delayed <- function(module, path, convert, delay_load) {
  
  .before_load <- delay_load$before_load %||% function() {}
  .on_load     <- delay_load$on_load %||% function() {}
  .on_error    <- delay_load$on_error %||% function(error) {}
  
  hooks <- list(
    
    before_load = function() {
      sys <- import("sys", convert = FALSE)
      sys$path$insert(0, path)
      .before_load()
    },
    
    on_load = function() {
      sys <- import("sys", convert = FALSE)
      sys$path$remove(path)
      .on_load()
    },
    
    on_error = function(error) {
      sys <- import("sys", convert = FALSE)
      sys$path$remove(path)
      .on_error(error)
    }
    
  )
  
  import(module, convert = convert, delay_load = hooks)
  
}

import_from_path_immediate <- function(module, path, convert) {
  
  # normalize path
  path <- normalizePath(path)
  
  # get sys module
  sys <- import("sys", convert = FALSE)
  
  # prepend the requested path, then remove it when we're done
  sys$path$insert(0L, path)
  on.exit(sys$path$remove(path), add = TRUE)
  
  # import the requested module
  import(module, convert = convert)
  
}




