
# prefer Python 3 if available
if (!is_windows() &&
    !py_available(initialize = FALSE) &&
    is.na(Sys.getenv("RETICULATE_PYTHON", unset = NA)))
{
  python <- Sys.which("python3")
  if (nzchar(python) && python != "/usr/bin/python3")
    use_python(python, required = TRUE)
}
