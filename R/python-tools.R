python_has_module <- function(python, module) {
  code <- paste("import", module)
  args <- c("-E", "-c", shQuote(code))
  status <- system2(python, args, stdout = FALSE, stderr = FALSE)
  status == 0L
}

python_version <- function(python) {
  code <- "import platform; print(platform.python_version())"
  args <- c("-E", "-c", shQuote(code))
  output <- system2(python, args, stdout = TRUE, stderr = TRUE)
  numeric_version(output)
}

python_module_version <- function(python, module) {
  fmt <- "import %1$s; print(%1$s.__version__)"
  code <- sprintf(fmt, module)
  args <- c("-E", "-c", shQuote(code))
  output <- system2(python, args, stdout = TRUE, stderr = TRUE)
  numeric_version(output)
}

python_unix_binary <- function(bin) {
  locations <- file.path(c("/usr/bin", "/usr/local/bin", path.expand("~/.local/bin")), bin)
  locations <- locations[file.exists(locations)]
  if (length(locations) > 0)
    locations[[1]]
  else
    NULL
}
