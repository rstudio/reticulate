
python_has_modules <- function(python, modules) {

  # write code to tempfile
  file <- tempfile("reticulate-python-", fileext = ".py")
  code <- paste("import", modules)
  writeLines(code, con = file)
  on.exit(unlink(file), add = TRUE)

  # invoke Python
  status <- system2(python, shQuote(file), stdout = FALSE, stderr = FALSE)
  status == 0L

}

python_has_module <- function(python, module) {
  code <- paste("import", module)
  args <- c("-E", "-c", shQuote(code))
  status <- system2(python, args, stdout = FALSE, stderr = FALSE)
  status == 0L
}

python_version <- function(python) {
  code <- "import platform; print(platform.python_version())"
  args <- c("-E", "-c", shQuote(code))
  output <- system2(python, args, stdout = TRUE, stderr = FALSE)
  sanitized <- gsub("[^0-9.-]", "", output)
  numeric_version(sanitized)
}

python_module_version <- function(python, module) {
  fmt <- "import %1$s; print(%1$s.__version__)"
  code <- sprintf(fmt, module)
  args <- c("-E", "-c", shQuote(code))
  output <- system2(python, args, stdout = TRUE, stderr = FALSE)
  numeric_version(output)
}

python_packages <- function(python) {
  args <- c("-m", "pip", "freeze")
  output <- system2(python, args, stdout = TRUE)
  splat <- strsplit(output, "==", fixed = TRUE)
  modules <- vapply(splat, `[[`, 1L, FUN.VALUE = character(1))
  versions <- vapply(splat, `[[`, 2L, FUN.VALUE = character(1))
  data.frame(module = modules, version = versions, stringsAsFactors = FALSE)
}

# given the path to a Python binary, try to ascertain its type
python_info <- function(python) {
  
  path <- dirname(python)
  parent <- dirname(path)
  
  while (path != parent) {
    
    # check for virtual environment files
    virtualenv <-
      file.exists(file.path(path, "pyvenv.cfg")) ||
      file.exists(file.path(path, ".Python"))

    if (virtualenv) {
      suffix <- if (is_windows()) "Scripts/python.exe" else "bin/python"
      python <- file.path(path, suffix)
      return(list(python = python, type = "virtualenv", root = path))
    }

    # check for conda-meta
    condaenv <-
      file.exists(file.path(path, "conda-meta")) &&
      !file.exists(file.path(path, "condabin"))

    if (condaenv) {
      suffix <- if (is_windows()) "python.exe" else "bin/python"
      python <- file.path(path, suffix)
      return(list(python = python, type = "conda", root = path))
    }
    
    # recurse
    parent <- path
    path <- dirname(path)
    
  }
  
  stopf("could not find a Python environment for %s", python)
  
}

