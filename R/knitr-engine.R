#' A reticulate Engine for Knitr
#' 
#' This provides a `reticulate` engine for `knitr`, suitable for usage when
#' attempting to render Python chunks. Using this engine allows for shared state
#' between Python chunks in a document -- that is, variables defined by one
#' Python chunk can be used by later Python chunks.
#' 
#' The engine can be activated by setting (for example)
#' 
#' ```
#' knitr::knit_engines$set(python = reticulate::eng_python)
#' ```
#' 
#' Typically, this will be set within a document's setup chunk, or by the
#' environment requesting that Python chunks be processed by this engine.
#' 
#' @param options
#'   Chunk options, as provided by `knitr` during chunk execution.
#'   
#' @export
eng_python <- function(options) {
  
  # extract the code to be run -- we'll attempt to run the code line by line
  # and detect changes so that we can interleave code and output (similar to
  # what one sees when executing an R chunk in knitr). to wit, we'll do our
  # best to emulate the return format of 'evaluate::evaluate()'
  code <- options$code
  if (!length(code))
    return(list())
  
  # helper function for extracting code
  extract <- function(code, start, end) {
    pasted <- paste(code[start:end], collapse = "\n")
    sub("^[\\r\\n\\t ]*\\n", "", pasted)
  }
  
  # indices tracking the current lines of code to be executed
  start <- end <- 1
  n <- length(code)
  
  # record where pending console output should be collected from
  pending <- 1
  
  # actual outputs to be returned to knitr
  outputs <- list()
  
  while (TRUE) {
    
    # check for end situation
    if (start > n || end > n)
      break
    
    # attempt evaluate of current chunk of code
    snippet <- extract(code, start, end)
    captured <- py_capture_output(
      result <- tryCatch(
        py_run_string(snippet, convert = FALSE),
        error = identity
      )
    )
    
    # detect syntax errors: assume that this means we need to pass an extra
    # line of input to reticulate
    #
    # TODO: it'd be nice if we could submit incomplete lines of code to
    # reticulate
    if (inherits(result, "error")) {
      message <- result$message
      if (is.character(message) && any(grepl("SyntaxError", message))) {
        end <- end + 1
        next
      }
    }
    
    # if we have console output, append that
    if (is.character(captured) && nzchar(captured)) {
      
      # append pending source to outputs
      outputs[[length(outputs) + 1]] <- structure(
        list(src = extract(code, pending, end)),
        class = "source"
      )
      
      # append captured outputs
      outputs[[length(outputs) + 1]] <- captured
      
      # TODO: find and append image data?
      
      # update pending state
      pending <- end + 1
    }
    
    # bump the start and end indices and continue
    end <- end + 1
    start <- end
  }
  
  # if we have leftover input, add that now
  if (pending < n) {
    leftover <- extract(code, pending, n)
    outputs[[length(outputs) + 1]] <- structure(
      list(src = leftover),
      class = "source"
    )
  }
  
  wrap <- yoink("knitr", "wrap")
  wrap(outputs, options)
  
}