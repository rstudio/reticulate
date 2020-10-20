
#' Scaffold R wrappers for Python functions
#'
#' @param python_function Fully qualified name of Python function or class
#'   constructor (e.g. `tf$layers$average_pooling1d`)
#' @param r_prefix Prefix to add to generated R function name
#' @param r_function Name of R function to generate (defaults to name of Python
#'   function if not specified)
#'
#' @note The generated wrapper will often require additional editing (e.g. to
#'   convert Python list literals in the docs to R lists, to massage R numeric
#'   values to Python integers via `as.integer` where required, etc.) so is
#'   really intended as an starting point for an R wrapper rather than a wrapper
#'   that can be used without modification.
#'
#' @keywords internal
#'
#' @export
py_function_wrapper <- function(python_function, r_prefix = NULL, r_function = NULL) {

  # get the docs
  docs <- py_function_docs(python_function)

  # generate wrapper w/ roxygen
  con <- textConnection("wrapper", "w")

  # title/description
  write(sprintf("#' %s\n#' ", docs$description), file = con)
  details <- gsub("\n", "\n#' ", docs$details, fixed = TRUE)
  write(sprintf("#' %s\n#' ", details), file = con)

  # parameters
  for (param in names(docs$parameters)) {
    # Note: the gsub is needed to prefix necessary indentations
    # to bullet points in parameter description
    param_description <- gsub("\n", "\n#'  ", docs$parameters[[param]], fixed = TRUE)
    write(sprintf("#' @param %s %s", param, param_description), file = con)
  }

  # returns
  if (isTRUE(nzchar(docs$returns))) {
    write("#' ", file = con)
    write(sprintf("#' @return %s", gsub("^\n", "", docs$returns)), file = con)
  }

  # sections
  for (section in names(docs$sections)) {
    section_text <- docs$sections[[section]]
    section_text <- gsub("\n", "\n#' ", section_text, fixed = TRUE)
    write("#' ", file = con)
    write(sprintf("#' @section %s:\n#' %s", section, section_text), file = con)
  }

  # export
  write("#' ", file = con)
  write("#' @export", file = con)

  # signature
  if (is.null(r_function)) {
    r_function <- docs$name
    if (!is.null(r_prefix))
      r_function <- paste(r_prefix, r_function, sep = "_")
  }
  signature <- sub(paste0(docs$name, "\\("), paste(r_function, "<- function("), docs$signature)
  write(paste(signature, "{"), file = con)

  # delegation
  write(paste0("  ", python_function, "("), file = con)
  params <- names(docs$parameters)
  if (length(params) > 0) {
    for (i in 1:length(params)) {
      suffix <- if (i < length(params))
        ","
      else
        "\n  )"
      write(paste0("    ", params[[i]], " = ", params[[i]], suffix), file = con)
    }
  } else {
    write(")", file = con)
  }

  # end function
  write("}", file = con)

  # close the connection
  close(con)

  # return the wrapper with a special class so we can write a print method
  class(wrapper) <- c("py_wrapper", "character")
  wrapper
}

#' @rdname py_function_wrapper
#' @export
py_function_docs <- function(python_function) {

  # eval so that python loads
  eval(parse(text = python_function))

  # get components
  components <- strsplit(python_function, "\\$")[[1]]
  topic <- components[[length(components)]]
  source <- paste(components[1:(length(components)-1)], collapse = "$")

  # get function docs
  function_docs <- help_handler(type = "completion", topic, source)

  # get parameter docs
  parameter_docs <- help_handler(type = "parameter", NULL, python_function)

  # create a named list with parameters
  parameters <- parameter_docs$arg_descriptions
  names(parameters) <- parameter_docs$args

  # create a new list with all doc info
  list(name = function_docs$title,
       qualified_name = python_function,
       description = function_docs$description,
       details = function_docs$details,
       signature = function_docs$signature,
       parameters = parameters,
       sections = function_docs$sections,
       returns = function_docs$returns)
}


#' @export
print.py_wrapper <- function(x, ...) {
  cat(x, sep = "\n")
}


#' Custom Scaffolding of R Wrappers for Python Functions
#'
#' This function can be used to generate R wrapper for a specified
#' Python function while allowing to inject custom code for critical parts of
#' the wrapper generation, such as process the any part of the docs obtained
#' from [py_function_docs()] and append additional roxygen fields. The result
#' from execution of `python_function` is assigned to a variable called
#' `python_function_result` that can also be processed by `postprocess_fn`
#' before writing the closing curly braces for the generated wrapper function.
#'
#' @inheritParams py_function_wrapper
#' @param additional_roxygen_fields A list of additional roxygen fields to write
#'   to the roxygen docs, e.g. `list(export = "", rdname =
#'   "generated-wrappers")`.
#' @param process_docs_fn A function to process docs obtained from
#'   `reticulate::py_function_docs(python_function)`.
#' @param process_param_fn A function to process each parameter needed for
#'   `python_funcion` before executing `python_funcion.`
#' @param process_param_doc_fn A function to process the roxygen docstring for
#'   each parameter.
#' @param postprocess_fn A function to inject any custom code in the form of a
#'   string before writing the closing curly braces for the generated wrapper
#'   function.
#' @param file_name The file name to write the generated wrapper function to. If
#'   `NULL`, the generated wrapper will only be printed out in the console.
#'
#' @examples
#' \dontrun{
#'
#' library(tensorflow)
#' library(stringr)
#'
#' # Example of a `process_param_fn` to cast parameters with default values
#' # that contains "L" to integers
#' process_int_param_fn <- function(param, docs) {
#'   # Extract the list of parameters that have integer values as default
#'   int_params <- gsub(
#'     " = [-]?[0-9]+L",
#'     "",
#'     str_extract_all(docs$signature, "[A-z]+ = [-]?[0-9]+L")[[1]])
#'   # Explicitly cast parameter in the list obtained above to integer
#'   if (param %in% int_params) {
#'     param <- paste0("as.integer(", param, ")")
#'   }
#'   param
#' }
#'
#' # Note that since the default value of parameter `k` is `1L`. It is wrapped
#' # by `as.integer()` to ensure it's casted to integer before sending it to `tf$nn$top_k`
#' # for execution. We then print out the python function result.
#' py_function_custom_scaffold(
#'   "tf$nn$top_k",
#'   r_function = "top_k",
#'   process_param_fn = process_int_param_fn,
#'   postprocess_fn = function() { "print(python_function_result)" })
#'
#' }
py_function_custom_scaffold <- function(
  python_function,
  r_function = NULL,
  additional_roxygen_fields = NULL,
  process_docs_fn = function(docs) docs,
  process_param_fn = function(param, docs) param,
  process_param_doc_fn = function(param_doc, docs) param_doc,
  postprocess_fn = function() {},
  file_name = NULL
) {

  write_line <- function(text) {
    cat(text, sep = "\n")
  }

  wrapper_output <- capture.output(
    {
      docs <- reticulate::py_function_docs(python_function)
      docs <- process_docs_fn(docs)

      write_line(paste0("#' @description ", docs$description))
      write_line(paste0("#' @title ", docs$name))
      write_line("#' ")

      # Write docstrings for each parameters
      for(i in 1:length(docs$parameters)) {
        param_name <- names(docs$parameters)[i]
        param_doc <- process_param_doc_fn(docs$parameters[[param_name]], docs)
        write_line(paste0("#' @param ", " ", names(docs$parameters)[i], " ", param_doc))
      }

      # Write additional roxygen fields
      if (!is.null(additional_roxygen_fields)) {
        if (!is.list(additional_roxygen_fields))
          stop("additional_roxygen_fields_list must be a list")
        invisible(lapply(names(additional_roxygen_fields), function(field_name) {
          write_line(paste0("#' @", field_name, " ", additional_roxygen_fields[[field_name]]))
        }))
      }

      # Change the name of the wrapper
      if (is.null(r_function)) {
        r_function <- docs$name
      }

      # Generate function signature
      signature <- sub(paste0(docs$name, "\\("),
                       paste(r_function, "<- function("), docs$signature)
      write_line(paste(signature, "{\n"))

      # Execute the Python function
      write_line(paste0("  python_function_result <- ", python_function, "("))

      # Generate parameters that get passed to the Python function call
      params <- names(docs$parameters)
      if (length(params) > 0) {
        for (i in 1:length(params)) {
          param <- params[i]
          suffix <- ifelse(i < length(params), ",", "\n  )")
          write_line(
            paste0(
              "    ",
              params[[i]], " = ",
              process_param_fn(param, docs),
              suffix))
        }
      } else {
        write_line(")")
      }

      # Inject additional custom code
      write_line(paste0("  ", postprocess_fn()))

      write_line("}\n")
    }
  )

  if (is.null(file_name)) {
    cat(wrapper_output, sep = "\n")
  } else {
    write(wrapper_output, file = file_name, append = TRUE)
  }
}

