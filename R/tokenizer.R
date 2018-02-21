python_tokenization_rules <- function() {
  
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
      pattern = "\"{3}(.*?)\"{3}",
      type    = "string"
    ),
    
    # consumes a '''string'''
    list(
      pattern = "'{3}(.*?)'{3}",
      type    = "string"
    ),
    
    # consumes a "string"
    list(
      pattern = '["](?:(?:\\\\.)|(?:[^"\\\\]))*?["]',
      type    = "string"
    ),
    
    # consumes a 'string'
    list(
      pattern = "['](?:(?:\\\\.)|(?:[^'\\\\]))*?[']",
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

# a simple regex-based tokenizer
tokenize <- function(text) {
  
  # vector of tokens
  tokens <- list()
  
  # rules to use
  rules <- python_tokenization_rules()
  
  # convert to raw vector so we can use 'grepRaw',
  # which supports offset-based search
  raw <- charToRaw(text)
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
      
      cat(sprintf("Token: [%s : %s]\n", rawToChar(match), rule$type))
      
      # update our vector of tokens
      tokens[[length(tokens) + 1]] <- list(
        token  = rawToChar(match),
        type   = rule$type,
        offset = offset
      )
      
      # update offset and break
      offset <- offset + length(match)
      break
      
    }
    
    # if we failed to match anything, consume a single character
    if (!matched) {
      
      tokens[[length(tokens) + 1]] <- list(
        token  = rawToChar(raw[[offset]]),
        type   = "unknown",
        offset = offset
      )
      
      offset <- offset + 1
      
    }
    
  }
  
  class(tokens) <- "tokens"
  tokens
  
}
