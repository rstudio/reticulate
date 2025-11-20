
# Methods for Populating the Positron/Ark Variables Pane
# These methods primarily delegate to the implementation in the
# Positron python_ipykernel.inspectors python module.


ark_positron_variable_display_value.python.builtin.object <- function(x, ..., width = getOption("width")) {
  val <- .globals$get_positron_variable_inspector(x)$get_display_value()[[1L]]
  substr(val, 1L, width)
}


ark_positron_variable_display_type.python.builtin.object <- function(x, ..., include_length = TRUE) {
  i <- .globals$get_positron_variable_inspector(x)
  out <-  i$get_display_type()

  if (startsWith(class(x)[1], "python.builtin.")) # display convert value?
    out <- paste("python", out)

  out
}


ark_positron_variable_kind.python.builtin.object <- function(x, ...) {
  i <- .globals$get_positron_variable_inspector(x)
  i$get_kind()
}


ark_positron_variable_has_children.python.builtin.object <- function(x, ...) {
  i <- .globals$get_positron_variable_inspector(x)
  i$has_children()
}


ark_positron_variable_get_children.python.builtin.object <- function(x, ...) {
  # Return an R list of children. The order of children should be
  # stable between repeated calls on the same object.
  i <- .globals$get_positron_variable_inspector(x)

  get_keys_and_children <- .globals$ark_variable_get_keys_and_children
  if (is.null(get_keys_and_children)) {
    get_keys_and_children <- .globals$ark_variable_get_keys_and_children <-
      import("rpytools.ark_variables", convert = FALSE)$get_keys_and_children
  }

  keys_and_children <- iterate(get_keys_and_children(i), simplify = FALSE)
  children <- iterate(keys_and_children[[2L]], simplify = FALSE)
  names(children) <- as.character(py_to_r(keys_and_children[[1L]]))

  children
}


ark_positron_variable_get_child_at.python.builtin.object <- function(x, ..., name, index) {
  i <- .globals$get_positron_variable_inspector(x)
  get_child <- .globals$ark_variable_get_child
  if (is.null(get_child)) {
    get_child <- .globals$ark_variable_get_child <-
      import("rpytools.ark_variables", convert = FALSE)$get_child
  }

  get_child(i, index)
}



ark_positron_variable_display_type.rpytools.ark_variables.ChildrenOverflow <-
  function(x, ..., include_length = TRUE) {
    ""
  }

ark_positron_variable_kind.rpytools.ark_variables.ChildrenOverflow <-
  function(x, ...) {
    "empty" # other? collection? map? lazy?
  }

ark_positron_variable_display_value.rpytools.ark_variables.ChildrenOverflow <-
  function(x, ..., width = getOption("width")) {
    paste(py_to_r(x$n_remaining), "more values")
  }

ark_positron_variable_has_children.rpytools.ark_variables.ChildrenOverflow <-
  function(x, ...) {
    FALSE
  }

ark_positron_help_get_handler.python.builtin.object <- local({
  positron_ipykernel <- NULL
  pydoc_thread <- NULL

  function(obj, ...) {
    if (is.null(positron_ipykernel)) {
      positron_ipykernel <<- import_from_path("positron", path = dirname(.globals$positron_ipykernel_path))
    }

    if (is.null(pydoc_thread)) {
      pydoc_thread <<- positron_ipykernel$pydoc$start_server(port = 64216L)
    }

    .ps.help.browse_external_url <- get(".ps.help.browse_external_url", globalenv())

    function(topic) {
      url <- paste0(
        pydoc_thread$url,
        "get?key=",
        positron_ipykernel$utils$get_qualname(obj)
      )

      .ps.help.browse_external_url(url)
      TRUE
    }
  }
})
  
