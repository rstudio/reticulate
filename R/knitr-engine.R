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
  
  ast <- import("ast")
  
  # helper function for extracting range of code, dropping blank lines
  extract <- function(code, range) {
    snippet <- code[range[1]:range[2]]
    snippet[nzchar(snippet)]
  }
  
  # extract the code to be run -- we'll attempt to run the code line by line
  # and detect changes so that we can interleave code and output (similar to
  # what one sees when executing an R chunk in knitr). to wit, we'll do our
  # best to emulate the return format of 'evaluate::evaluate()'
  code <- options$code
  n <- length(code)
  if (n == 0)
    return(list())
  
  # use 'ast.parse()' to parse Python code and collect line numbers, so we
  # can split source code into statements
  pasted <- paste(code, collapse = "\n")
  parsed <- ast$parse(pasted, "<string>")
  
  # iterate over top-level nodes and extract line numbers
  lines <- vapply(parsed$body, function(node) {
    node$lineno
  }, integer(1))
  
  # convert from lines to ranges
  starts <- lines
  ends <- c(lines[-1] - 1, length(code))
  ranges <- mapply(c, starts, ends, SIMPLIFY = FALSE)
  
  # line index from which source should be emitted
  pending <- 1
  
  # actual outputs to be returned to knitr
  outputs <- list()
  
  for (range in ranges) {
    
    # evaluate current chunk of code
    snippet <- extract(code, range)
    captured <- py_capture_output(
      py_run_string(snippet, convert = FALSE)
    )
    
    if (is.character(captured) && nzchar(captured)) {
      
      # append pending source to outputs
      outputs[[length(outputs) + 1]] <- structure(
        list(src = extract(code, c(pending, range[2]))),
        class = "source"
      )
      
      # append captured outputs
      outputs[[length(outputs) + 1]] <- captured
      
      # TODO: find and append image data?
      
      # update pending source range
      pending <- range[2] + 1
    }
  }
  
  # if we have leftover input, add that now
  if (pending < n) {
    leftover <- extract(code, c(pending, n))
    outputs[[length(outputs) + 1]] <- structure(
      list(src = leftover),
      class = "source"
    )
  }
  
  # TODO: development version of knitr supplies new 'engine_output()'
  # interface -- use that when it's on CRAN
  # https://github.com/yihui/knitr/commit/71bfd8796d485ed7bb9db0920acdf02464b3df9a
  wrap <- yoink("knitr", "wrap")
  wrap(outputs, options)
  
}