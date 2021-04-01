
case <- function(...) {
  
  dots <- list(...)
  for (dot in dots) {
    
    if (!inherits(dot, "formula"))
      return(dot)
    
    else if (length(dot) == 2) {
      expr <- dot[[2]]
      return(eval(expr, envir = environment(dot)))
    }
    
    else {
      
      cond <- dot[[2]]
      expr <- dot[[3]]
      if (eval(cond, envir = environment(dot)))
        return(eval(expr, envir = environment(dot)))
      
    }
  }
  
  NULL
  
}
