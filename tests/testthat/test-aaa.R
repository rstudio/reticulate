
# prefer Python 3 if available
if (!py_available(initialize = FALSE) &&
    is.na(Sys.getenv("RETICULATE_PYTHON", unset = NA)))
{
  python <- Sys.which("python3")
  if (nzchar(python))
    use_python(python, required = TRUE)
}

print(py_config())

