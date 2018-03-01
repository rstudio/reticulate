py_tokenization_rules <- function() {
  
  # we use the 'tokenize' module to provide a regular expression
  # for consuming Python numbers
  tokenize <- import("tokenize", convert = TRUE)
  
  list(
    
    # consume an identifier
    list(
      pattern = "[[:alpha:]_][[:alnum:]_]*",
      type    = "identifier"
    ),
    
    # consume a number
    list(
      pattern = tokenize$Number,
      type    = "number"
    ),
    
    # consumes a """string"""
    list(
      pattern = '["]{3}(.*?)(?:["]{3}|$)',
      type    = "string"
    ),
    
    # consumes a '''string'''
    list(
      pattern = "[']{3}(.*?)(?:[']{3}|$)",
      type    = "string"
    ),
    
    # consumes a "string"
    list(
      pattern = '["](?:(?:\\\\.)|(?:[^"\\\\]))*?(?:["]|$)',
      type    = "string"
    ),
    
    # consumes a 'string'
    list(
      pattern = "['](?:(?:\\\\.)|(?:[^'\\\\]))*?(?:[']|$)",
      type    = "string"
    ),
    
    # consume an operator
    list(
      pattern = "\\*\\*=?|>>=?|<<=?|<>|!+|//=?|[%&|^=<>*/+-]=?|~",
      type    = "operator"
    ),
    
    # consume a 'special' token
    list(
      pattern = "[:;.,`@]",
      type    = "special"
    ),
    
    # consumes a bracket
    list(
      pattern = "[][)(}{]",
      type    = "bracket"
    ),
    
    # consumes a comment
    list(
      pattern = "#[^\n]*",
      type    = "comment"
    ),
    
    # consumes whitespace
    list(
      pattern = "[[:space:]]+",
      type    = "whitespace"
    )
    
  )
  
}

py_token <- function(value, type, offset) {
  
  list(
    value  = value,
    type   = type,
    offset = offset
  )
  
}

# a simple regex-based tokenizer. does not understand all of the Python
# language; just enough to power the completion engine
py_tokenize <- function(
    code,
    exclude = function(token) FALSE,
    keep.unknown = TRUE)
{
  
  # vector of tokens
  tokens <- list()
  
  # rules to use
  rules <- py_tokenization_rules()
  
  # convert to raw vector so we can use 'grepRaw',
  # which supports offset-based search
  raw <- charToRaw(code)
  n <- length(raw)
  
  # record current offset
  offset <- 1
  
  while (offset <= n) {
  
    # record whether we successfully matched a rule
    matched <- FALSE
    
    # iterate through rules, looking for a match
    for (rule in rules) {
      
      # augment pattern to search only from start of requested offset
      pattern <- paste("^(?:", rule$pattern, ")", sep = "")
      match <- grepRaw(pattern, raw, offset = offset, value = TRUE)
      if (length(match) == 0)
        next
      
      # we found a match; record that
      matched <- TRUE
      
      # update our vector of tokens
      token <- py_token(rawToChar(match), rule$type, offset)
      if (!exclude(token))
        tokens[[length(tokens) + 1]] <- token
      
      # update offset and break
      offset <- offset + length(match)
      break
      
    }
    
    # if we failed to match anything, consume a single character
    if (!matched) {
      # update tokens
      token <- py_token(rawToChar(raw[[offset]]), "unknown", offset)
      if (keep.unknown && !exclude(token))
        tokens[[length(tokens) + 1]] <- token
      
      # update offset
      offset <- offset + 1
    }
    
  }
  
  class(tokens) <- "tokens"
  tokens
  
}

# a helper object for moving around tokens
py_token_cursor <- function(tokens) {
  
  .tokens <- tokens
  .offset <- 1L
  .n <- length(tokens)
  
  .lbrackets <- c("(", "{", "[")
  .rbrackets <- c(")", "}", "]")
  .complements <- list(
    "(" = ")", "[" = "]", "{" = "}",
    ")" = "(", "]" = "[", "}" = "{"
  )
  
  tokenValue   <- function() { .tokens[[.offset]]$value  }
  tokenType    <- function() { .tokens[[.offset]]$type   }
  tokenOffset  <- function() { .tokens[[.offset]]$offset }
  cursorOffset <- function() { .offset                   }
  
  moveToOffset <- function(offset) {
    if (offset < 1L)
      .offset <<- 1L
    else if (offset > .n)
      .offset <<- .n
    else
      .offset <<- offset
  }
  
  moveToNextToken <- function(i = 1L) {
    offset <- .offset + i
    if (offset > .n)
      return(FALSE)
    
    .offset <<- offset
    return(TRUE)
  }
  
  moveToPreviousToken <- function(i = 1L) {
    offset <- .offset - i
    if (offset < 1L)
      return(FALSE)
    
    .offset <<- offset
    return(TRUE)
  }
  
  moveRelative <- function(i = 1L) {
    offset <- .offset + i
    if (offset < 1L || offset > .n)
      return(FALSE)
    
    .offset <<- offset
    return(TRUE)
  }
  
  fwdToMatchingBracket <- function() {
    
    token <- .tokens[[.offset]]
    value <- token$value
    if (!value %in% .lbrackets)
      return(FALSE)
    
    lhs <- value
    rhs <- .complements[[lhs]]
    
    count <- 1
    while (moveToNextToken()) {
      value <- tokenValue()
      if (value == lhs) {
        count <- count + 1
      } else if (value == rhs) {
        count <- count - 1
        if (count == 0)
          return(TRUE)
      }
    }
    
    return(FALSE)
  }
  
  bwdToMatchingBracket <- function() {
    
    token <- .tokens[[.offset]]
    value <- token$value
    if (!value %in% .rbrackets)
      return(FALSE)
    
    lhs <- value
    rhs <- .complements[[lhs]]
    
    count <- 1
    while (moveToPreviousToken()) {
      value <- tokenValue()
      if (value == lhs) {
        count <- count + 1
      } else if (value == rhs) {
        count <- count - 1
        if (count == 0)
          return(TRUE)
      }
    }
    
    return(FALSE)
  }
  
  peek <- function(i = 0L) {
    offset <- .offset + i
    if (offset < 1L || offset > .n)
      return(list(token = "", type = "unknown", offset = offset))
    return(.tokens[[offset]])
  }
  
  find <- function(predicate, forward = TRUE) {
    if (forward) {
      offset <- .offset + 1L
      while (offset <= .n) {
        token <- .tokens[[offset]]
        if (predicate(token)) {
          .offset <<- offset
          return(TRUE)
        }
        offset <- offset + 1L
      }
      return(FALSE)
    } else {
      offset <- .offset - 1L
      while (offset >= 1L) {
        token <- .tokens[[offset]]
        if (predicate(token)) {
          .offset <<- offset
          return(TRUE)
        }
        offset <- offset - 1L
      }
      return(FALSE)
    }
  }
  
  # move to the start of a Python statement, e.g.
  #
  #    alpha.beta["gamma"]
  #    ^~~~~~~~~<~~~~~~~~^
  #
  moveToStartOfEvaluation <- function() {
    
    repeat {
      
      # skip matching brackets
      if (bwdToMatchingBracket()) {
        if (!moveToPreviousToken())
          return(TRUE)
        next
      }
      
      # if the previous token is an identifier or a '.', move on to it
      previous <- peek(-1L)
      if (previous$value %in% "." || previous$type %in% "identifier") {
        moveToPreviousToken()
        next
      }
      
      break
      
    }
    
    TRUE
  }
  
  list(
    tokenValue              = tokenValue,
    tokenType               = tokenType,
    tokenOffset             = tokenOffset,
    cursorOffset            = cursorOffset,
    moveToNextToken         = moveToNextToken,
    moveToPreviousToken     = moveToPreviousToken,
    fwdToMatchingBracket    = fwdToMatchingBracket,
    bwdToMatchingBracket    = bwdToMatchingBracket,
    moveToOffset            = moveToOffset,
    moveRelative            = moveRelative,
    peek                    = peek,
    find                    = find,
    moveToStartOfEvaluation = moveToStartOfEvaluation
  )
  
}
