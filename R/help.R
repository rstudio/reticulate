

#' Documentation for Python Objects
#'
#' @param object Object to print documentation for
#'
#' @export
py_help <- function(object) {
  help <- py_capture_output(import_builtins()$help(object), type = "stdout")
  tmp <- tempfile("py_help", fileext = ".txt")
  writeLines(help, con = tmp)
  file.show(tmp, title = paste("Python Help:", object$`__name__`), delete.file = TRUE)
}

#' Register help topics
#'
#' Register a set of help topics for dispatching from F1 help
#'
#' @param type Type (module or class)
#' @param topics Named list of topics
#'
#' @keywords internal
#' @export
register_help_topics <- function(type = c("module", "class"), topics) {

  # pick the right environment for this type
  type <- match.arg(type)
  envir <- switch(type,
                  module = .module_help_topics,
                  class = .class_help_topics
  )

  # assign the list into the environment
  for (name in names(topics))
    assign(name, topics[[name]], envir = envir)
}

# Generic help_handler returned from .DollarNames -- dispatches to various
# other help handler functions
help_handler <- function(type = c("completion", "parameter", "url"), topic, source, ...) {
  type <- match.arg(type)
  if (type == "completion") {
    help_completion_handler.python.builtin.object(topic, source)
  } else if (type == "parameter") {
    help_completion_parameter_handler.python.builtin.object(source)
  } else if (type == "url") {
    help_url_handler.python.builtin.object(topic, source)
  }
}

#' Register a help handler for a root Python module
#'
#' @param module Name of a root Python module
#' @param handler Handler function: `function(name, subtopic = NULL)`. The name
#'   will be the fully qualfied name of a Python object (module, function, or
#'   class). The `subtopic` is optional and is currently used only for methods
#'   within classes.
#'
#' @details The help handler is passed a fully qualfied module, class, or
#'   function name (and optional method for classes). It should return a URL for
#'   a help page (or `""` if no help page is available).
#'
#' @keywords internal
#' @export
register_module_help_handler <- function(module, handler) {
  .module_help_handlers[[module]] <- handler
}

# Extract docs in TensorFlow-like styles
help_completion_handler_default <- function(doc) {
  arguments_matches <- regexpr(pattern = '\n(Arg(s|uments):)', doc)
  if (arguments_matches[[1]] != -1)
    description <- substring(doc, 1, arguments_matches[[1]])
  else
    description <- doc

  # collect other sections
  sections <- sections_from_doc(doc)

  # try to get return info
  returns <- sections$Returns

  # remove arguments and returns
  sections$Args <- NULL
  sections$Arguments <- NULL
  sections$Returns <- NULL

  list(
    description = description,
    sections = sections,
    returns = returns
  )
}

# Extract docs in Sphinx style
help_completion_handler_sphinx <- function(doc) {
  doctree <- sphinx_doctree_from_doc(doc)
  returns <- gsub("Returns\n", "", doctree$ids$returns$astext(), fixed = TRUE)
  # remove the additional space before ":" as it's Sphinx specific
  returns <- gsub(" : ", ": ", returns, fixed = TRUE)
  description <- substring(doc, 1, sphinx_doc_params_matches(doc)[[1]])
  # extract sections other than parameters and returns
  sections <- lapply(names(doctree$ids), function(name)
    if (!name %in% c("parameters", "returns")) doctree$ids[[name]])
  sections[sapply(sections, is.null)] <- NULL

  list(
    description = description,
    sections = sections,
    returns = returns
  )
}

# Return help for display in the completion popup window
help_completion_handler.python.builtin.object <- function(topic, source) {

  if (!py_available())
    return(NULL)

  # convert source to object if necessary
  source <- source_as_object(source)
  if (is.null(source))
    return(NULL)

  # check for property help
  help <- import("rpytools.help")
  doc <- help$get_property_doc(source, topic)
  # check for standard help
  if (is.null(doc)) {
    inspect <- import("inspect")
    doc <- inspect$getdoc(help_get_attribute(source, topic))
  }
  # default to no doc
  if (is.null(doc))
    doc <- ""

  if (is_sphinx_doc(doc) && is_docutils_available()) {
    extracted <- tryCatch(
      help_completion_handler_sphinx(doc),
      error = identity)
    if (inherits(extracted, "error"))
      extracted <- help_completion_handler_default(doc)
  } else {
    extracted <- help_completion_handler_default(doc)
  }

  description <- extracted$description
  sections <- extracted$sections
  returns <- extracted$returns

  # extract description and details
  matches <- regexpr(pattern ='\n', description, fixed=TRUE)
  if (matches[[1]] != -1) {
    details <- substring(description, matches[[1]] + 1)
    description <- substring(description, 1, matches[[1]] - 1)
  } else {
    details <- ""
  }
  details <- cleanup_description(details)
  description <- cleanup_description(description)

  # try to generate a signature
  signature <- NULL
  target <- help_get_attribute(source, topic)
  if (!is.null(target) && py_is_callable(target)) {
    help <- import("rpytools.help")
    signature <- help$generate_signature_for_function(target)
    if (is.null(signature))
      signature <- "()"
    signature <- paste0(topic, signature)
  }

  # return docs
  list(title = topic,
       signature = signature,
       returns = returns,
       description = description,
       details = details,
       sections = sections)
}


# Return parameter help for display in the completion popup window
help_completion_parameter_handler.python.builtin.object <- function(source) {

  if (!py_available())
    return(NULL)

  # split into topic and source
  components <- source_components(source)
  if (is.null(components))
    return(NULL)
  topic <- components$topic
  source <- components$source

  # get the function
  target <- help_get_attribute(source, topic)
  if (!is.null(target) & py_is_callable(target)) {
    help <- import("rpytools.help")
    args <- help$get_arguments(target)
    if (!is.null(args)) {
      # get the descriptions
      doc <- help$get_doc(target)
      arg_descriptions <- arg_descriptions_from_doc(args, doc)
      return(list(
        args = args,
        arg_descriptions = arg_descriptions
      ))
    }
  }

  # no parameter help found
  NULL
}


# Handle requests for external (F1) help
help_url_handler.python.builtin.object <- function(topic, source) {

  if (!py_available())
    return(NULL)

  # normalize topic and source for various calling scenarios
  if (grepl(" = $", topic)) {
    components <- source_components(source)
    if (is.null(components))
      return(NULL)
    topic <- components$topic
    source <- components$source
  } else {
    source <- source_as_object(source)
    if (is.null(source))
      return(NULL)
  }

  # get help page
  page <- NULL
  inspect <- import("inspect")
  if (inspect$ismodule(source)) {
    module <- py_get_name(source)
    help <- module_help(module, topic)
  } else {
    help <- class_help(class(source), topic)
  }

  # return help (can be "")
  help
}


# Handle requests for the list of arguments for a function
help_formals_handler.python.builtin.object <- function(topic, source) {

  if (!py_available())
    return(NULL)

  # check for module proxy
  if (py_is_module_proxy(source))
    return(NULL)

  if (py_has_attr(source, topic)) {
    target <- help_get_attribute(source, topic)
    if (!is.null(target) && py_is_callable(target)) {
      help <- import("rpytools.help")
      args <- help$get_arguments(target)
      if (!is.null(args)) {
        return(list(
          formals = args,
          helpHandler = "reticulate:::help_handler"
        ))
      }
    }
  }

  # default to NULL if we couldn't get the arguments
  NULL
}

sphinx_doc_params_matches <- function(doc) {
  regexpr(pattern = "\nParameters\n+[-]+", doc)
}

is_sphinx_doc <- function(doc) {
  sphinx_doc_params_matches(doc)[[1]] != -1
}

is_docutils_available <- function() {
  py_module_available("docutils")
}

sphinx_doctree_from_doc <- function(doc) {
  docutils <- import("docutils")
  py_capture_output(
    doctree <- docutils$core$publish_doctree(doc)
  )
  doctree
}

# Extract arguments descriptions for docs in TensorFlow-like styles
arg_descriptions_from_doc_default <- function(args, doc) {
  # extract arguments section of the doc and break into lines
  arguments <- section_from_doc('Arg(s|uments)', doc)
  doc <- strsplit(doc, "\n", fixed = TRUE)[[1]]

  sapply(args, function(arg) {
    arg_line <- which(grepl(paste0("^\\s+", arg, ":"), doc))
    if (length(arg_line) > 0) {
      line <- doc[[arg_line]]
      arg_description <- substring(line, regexpr(':', line)[[1]] + 1)
      next_line <- arg_line + 1
      while((arg_line + 1) <= length(doc)) {
        line <- doc[[arg_line + 1]]
        if (!grepl("^\\s*$", line) && !grepl("^\\s+\\w+: ", line)) {
          arg_description <- paste(arg_description, line)
          arg_line <- arg_line + 1
        }
        else
          break
      }
      arg_description <- cleanup_description(arg_description)
    } else {
      arg
    }
  })
}

# Extract arguments descriptions for docs in Sphinx style
arg_descriptions_from_doc_sphinx <- function(doc) {
  doctree <- sphinx_doctree_from_doc(doc)
  params <- doctree$ids$parameters$children[[2]]$children

  text <- vapply(params, function(param) {
    param$children[[3]]$astext()
  }, character(1))

  nm <- vapply(params, function(param) {
    param$children[[1]]$astext()
  }, character(1))

  names(text) <- nm
  text
}

# Extract argument descriptions from python docstring
arg_descriptions_from_doc <- function(args, doc) {
  if (is.null(doc)) {
    arg_descriptions <- args
  } else if (is_sphinx_doc(doc) && is_docutils_available()) {
    arg_descriptions <- tryCatch(
      arg_descriptions_from_doc_sphinx(doc),
      error = identity)
    if (inherits(arg_descriptions, "error"))
      arg_descriptions <- arg_descriptions_from_doc_default(args, doc)
  } else {
    arg_descriptions <- arg_descriptions_from_doc_default(args, doc)
  }
  arg_descriptions
}

# Extract all sections from the doc
sections_from_doc <- function(doc) {

  # sections to return
  sections <- list()

  # grab section headers
  doc <- strsplit(doc, "\n", fixed = TRUE)[[1]]
  section_lines <- which(grepl("^\\w(\\w|\\s)+:", doc))

  # for each section
  for (i in section_lines) {

    # get the section line and name
    section_line <- i
    section_name <- gsub(":\\s*$", "", doc[[i]])

    # collect the sections text
    section_text <- c()
    while((section_line + 1) <= length(doc)) {
      line <- doc[[section_line + 1]]
      if (grepl("\\w+", line)) {
        section_text <- paste(section_text, line)
        section_line <- section_line + 1
      }
      else
        break
    }

    # add to our list
    sections[[section_name]] <- cleanup_description(section_text)
  }

  # return the sections
  sections
}


# Extract section from doc
section_from_doc <- function(name, doc) {
  section <- ""
  doc <- strsplit(doc, "\n", fixed = TRUE)[[1]]
  line_index <- which(grepl(paste0("^", name, ":"), doc))
  if (length(line_index) > 0) {
    while((line_index + 1) <= length(doc)) {
      line <- doc[[line_index + 1]]
      if (grepl("\\w+", line)) {
        section <- paste(section, line)
        line_index <- line_index + 1
      }
      else
        break
    }
  }
  cleanup_description(section)
}

# Convert types in description
cleanup_description <- function(description) {
  if (is.null(description)) {
    NULL
  } else {

    # remove leading and trailing whitespace
    description <- gsub("^\\s+|\\s+$", "", description)

    # convert 2+ whitespace to 1 ws
    description <- gsub("(\\s\\s+)", " ", description)

    # convert literals
    description <- gsub("None", "NULL", description)
    description <- gsub("True", "TRUE", description)
    description <- gsub("False", "FALSE", description)

    # convert tuple to list
    description <- gsub("tuple", "list", description)
    description <- gsub("list/list", "list", description)

    description
  }
}

# Convert source to object if necessary
source_as_object <- function(source) {

  if (is.character(source)) {
    source <- tryCatch(eval(parse(text = source), envir = globalenv()),
                       error = function(e) NULL)
    if (is.null(source))
      return(NULL)
  }

  source
}

# Split source string into source and topic
source_components <- function(source) {
  components <- strsplit(source, "\\$")[[1]]
  topic <- components[[length(components)]]
  source <- paste(components[1:(length(components)-1)], collapse = "$")
  source <- source_as_object(source)
  if (!is.null(source))
    list(topic = topic, source = source)
  else
    NULL
}


module_help <- function(module, topic) {

  # do we have a page for this module/topic?
  lookup <- paste(module, topic, sep = ".")
  page <- .module_help_topics[[lookup]]

  # if so then append topic and return
  if (!is.null(page))
    return(paste(page, topic, sep = "#"))

  # do we have a module handler
  main_module <- strsplit(module, ".", fixed = TRUE)[[1]][[1]]
  handler <- .module_help_handlers[[main_module]]
  if (!is.null(handler))
    handler(lookup)
  else
    ""
}

class_help <- function(class, topic) {

  # call recursively for more than one class
  if (length(class) > 1) {
    # call for each class
    for (i in 1:length(class)) {
      help <- class_help(class[[i]], topic)
      if (nzchar(help))
        return(help)
    }
    # no help found
    return("")
  }

  # do we have a page for this class?
  page <- .class_help_topics[[class]]

  # if so then append class and topic and return
  if (!is.null(page)) {
    components <- strsplit(class, ".", fixed = TRUE)[[1]]
    class <- components[[length(components)]]
    return(paste0(page, "#", class, ".", topic))
  }

  # do we have a handler for this module
  main_module <- strsplit(class, ".", fixed = TRUE)[[1]][[1]]
  handler <- .module_help_handlers[[main_module]]
  if (!is.null(handler))
    handler(class, topic)
  else
    ""
}

help_get_attribute <- function(source, topic) {

  # check for module proxy
  if (py_is_module_proxy(source))
    return(NULL)

  # check for sub-module
  if (py_is_module(source) && !py_has_attr(source, topic)) {
    module <- py_get_submodule(source, topic)
    if (!is.null(module))
      return(module)
  }

  # get attribute w/ no warnings or errors
  tryCatch(py_suppress_warnings(py_get_attr(source, topic)),
           error = clear_error_handler(NULL))
}

# Environments where we store help topics (mappings of module/class name to URL)
.module_help_topics <- new.env(parent = emptyenv())
.class_help_topics <- new.env(parent = emptyenv())
.module_help_handlers <- new.env(parent = emptyenv())


