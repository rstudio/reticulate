
# prefer Python 3 if available
if (!is_windows() &&
    !py_available(initialize = FALSE) &&
    is.na(Sys.getenv("RETICULATE_PYTHON", unset = NA)))
{
  python <- Sys.which("python3")
  if (nzchar(python))
    use_python(python, required = TRUE)
}

# TODO: Install Python on CI for Windows
if (!is_windows())
    print(py_config())

