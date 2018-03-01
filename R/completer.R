py_completions <- function(token, candidates) {
  pattern <- paste("^\\Q", token, "\\E", sep = "")
  completions <- sort(grep(pattern, candidates, perl = TRUE, value = TRUE))
  attr(completions, "token") <- token
  completions
}

py_complete_none <- function() {
  character()
}

# retrieve module completions
py_complete_imports <- function(token) {
  modules <- py_list_modules()
  
  # check to see if the user is attempting to import a submodule
  dots <- gregexpr(".", token, fixed = TRUE)[[1]]
  if (identical(c(dots), -1L)) {
    toplevel <- grep(".", modules, fixed = TRUE, invert = TRUE, value = TRUE)
    return(py_completions(token, toplevel))
  }
  
  # we had dots; list all sub-modules of that module (since
  # we're now scoped to sub-modules it doesn't hurt to
  # display potentially more than just one layer)
  py_completions(token, modules)
}

py_complete_files <- function(token) {
  
  os <- import("os", convert = TRUE)
  token <- gsub("^['\"]|['\"]$", "", token)
  expanded <- path.expand(token)
  
  # for compatibility with R + readline, we use the R completion system
  # so that file completions are processed as expected there
  if (!is_rstudio()) {
    utils <- asNamespace("utils")
    completions <- tryCatch(utils$fileCompletions(expanded), error = identity)
    if (inherits(completions, "error"))
      return(character())
    return(completions)
  }
  
  # find the index of the last slash -- everything following is
  # the completion token; everything before is the directory to
  # search for completions in
  indices <- gregexpr("/", expanded, fixed = TRUE)[[1]]
  if (!identical(c(indices), -1L)) {
    lhs <- substring(expanded, 1, tail(indices, n = 1))
    rhs <- substring(expanded, tail(indices, n = 1) + 1)
    files <- paste(lhs, list.files(lhs), sep = "")
  } else {
    lhs <- "."
    rhs <- expanded
    files <- list.files(os$getcwd())
  }
  
  # form completions (but add extra metadata after)
  completions <- py_completions(expanded, files)
  attr(completions, "token") <- token
  
  info <- file.info(completions)
  attr(completions, "types") <- ifelse(info$isdir, 16, 15)
  
  completions
}

py_complete_keys <- function(object, token) {
  
  method <- py_get_attr(object, "keys", silent = TRUE)
  if (!inherits(method, "python.builtin.object"))
    return(py_complete_none())
  
  keys <- py_to_r(method)
  candidates <- py_to_r(keys())
  py_completions(token, candidates)
}

py_complete_functions <- function(object, token) {
  
  # use 'inspect' module to grab function arguments
  inspect <- import("inspect", convert = TRUE)
  arguments <- inspect$getargspec(object)$args
  
  # paste on an '=' for completions (Python users seem to prefer no
  # spaces between the argument name and value)
  completions <- paste(arguments, "=", sep = "")
  
  py_completions(token, completions)
  
}

py_complete_default <- function(token) {
  
  dots <- gregexpr(".", token, fixed = TRUE)[[1]]
  if (identical(c(dots), -1L)) {
    # no dots; just try to complete objects from the main module
    main     <- import_main(convert = TRUE)
    builtins <- import_builtins(convert = TRUE)
    keyword  <- import("keyword", convert = TRUE)
    
    candidates <- c(names(main), names(builtins), keyword$kwlist)
    return(py_completions(token, candidates))
  }

  # we had dots; try to evaluate the left-hand side of the dots
  # and then filter on the attributes of the object (if any)
  last <- tail(dots, n = 1)
  lhs <- substring(token, 1, last - 1)
  rhs <- substring(token, last + 1)

  # try evaluating the left-hand side
  object <- tryCatch(py_eval(lhs, convert = FALSE), error = identity)
  if (inherits(object, "error"))
    return(py_complete_none())

  # attempt to get completions
  candidates <- tryCatch(py_list_attributes(object), error = identity)
  if (inherits(candidates, "error"))
    return(py_complete_none())
  
  # R readline completion, and older versions of RStudio,
  # require us to keep the '.'s (newer versions of RStudio will
  # trim the '.' and display the completions more neatly)
  rhs <- paste(lhs, rhs, sep = ".")
  candidates <- paste(lhs, candidates, sep = ".")
  
  py_completions(rhs, candidates)
}

# basic autocompletion support (used for the Python REPL)
py_completer <- function(line) {
  
  # check for completion of a module name in e.g. 'import nu'
  re_import <- "^[[:space:]]*import[[:space:]]+([[:alnum:]._]*)$"
  matches <- regmatches(line, regexec(re_import, line, perl = TRUE))[[1]]
  if (length(matches) == 2)
    return(py_complete_imports(matches[[2]]))
  
  # tokenize the line and grab the last token
  tokens <- py_tokenize(
    code = line,
    exclude = function(token) { token$type %in% c("whitespace", "comment") },
    keep.unknown = FALSE
  )
  
  if (length(tokens) == 0)
    return(py_complete_none())

  # construct token cursor
  cursor <- py_token_cursor(tokens)
  cursor$moveToOffset(length(tokens))
  token <- cursor$peek()
  
  # for strings, we may be either completing dictionary keys or files
  if (token$type %in% "string") {
    
    # if there's no prior token, assume this is a file name
    if (!cursor$moveToPreviousToken())
      return(py_complete_files(token$value))
    
    # if the prior token is an open bracket, assume we're completing
    # a dictionary key
    if (cursor$tokenValue() == "[") {
      
      saved <- cursor$peek()
      
      if (!cursor$moveToPreviousToken())
        return(py_complete_none())
      
      if (!cursor$moveToStartOfEvaluation())
        return(py_complete_none())
      
      # grab text from this offset
      lhs <- substring(line, cursor$tokenOffset(), saved$offset - 1)
      rhs <- gsub("^['\"]|['\"]$", "", token$value)
      
      # bail if there are any '(' tokens (avoid arbitrary function eval)
      # in theory this screens out tuples but that's okay for now
      tokens <- py_tokenize(lhs)
      lparen <- Find(function(token) token$value == "(", tokens)
      if (!is.null(lparen))
        return(py_complete_none())
      
      # attempt to evaluate left-hand side
      evaluated <- tryCatch(py_eval(lhs, convert = FALSE), error = identity)
      if (inherits(evaluated, "error"))
        return(py_complete_none())
      
      return(py_complete_keys(evaluated, rhs))
      
    }
    
    # doesn't look like a dictionary; perform filesystem completion
    return(py_complete_files(token$value))
    
  }
  
  # try to guess if we're trying to autocomplete function arguments
  maybe_function <-
    cursor$peek(0 )$value %in% c("(", ",") ||
    cursor$peek(-1)$value %in% c("(", ",")
  
  if (maybe_function) {
    offset <- cursor$cursorOffset()
    
    # try to find an opening bracket
    repeat {
      
      # skip matching brackets
      if (cursor$bwdToMatchingBracket()) {
        if (!cursor$moveToPreviousToken())
          return(py_complete_none())
        next
      }
      
      # if we find an opening bracket, check to see if the token to the
      # left is something that is, or could produce, a function
      if (cursor$tokenValue() == "(" &&
          cursor$moveToPreviousToken() &&
          (cursor$tokenValue() == "]" || cursor$tokenType() %in% "identifier"))
      {
        # find code to be evaluted that will produce function
        endToken   <- cursor$peek()
        cursor$moveToStartOfEvaluation()
        startToken <- cursor$peek()
        
        # extract the associated text
        start <- startToken$offset
        end   <- endToken$offset + nchar(endToken$value) - 1
        text <- substring(line, start, end)
        
        # attempt to evaluate it
        object <- tryCatch(py_eval(text, convert = FALSE), error = identity)
        if (inherits(object, "error"))
          break
        
        # success! get argument completions
        rhs <- if (token$type %in% "identifier") token$value else ""
        return(py_complete_arguments(object, rhs))
      }
      
      if (!cursor$moveToPreviousToken())
        break
    }
    
    # if we got here, our attempts to find a function failed, so
    # go home and fall back to the default completion solution
    cursor$moveToOffset(offset)
  }
  
  # start looking backwards
  repeat {
    
    # skip matching brackets
    if (cursor$bwdToMatchingBracket()) {
      if (!cursor$moveToPreviousToken())
        return(py_complete_none())
      next
    }
    
    # consume identifiers, strings, '.'
    if (cursor$tokenType() %in% c("string", "identifier") ||
        cursor$tokenValue() %in% ".")
    {
      # if we can't move to the previous token, we must be at the
      # start of the token stream, so just consume from here
      if (!cursor$moveToPreviousToken())
        break
      next
    }
    
    # if this isn't a matched token, then move back up a single
    # token and break
    if (!cursor$moveToNextToken())
      return(py_complete_none())
    
    break
    
  }
  
  text <- substring(line, cursor$tokenOffset())
  py_complete_default(text)
  
}
