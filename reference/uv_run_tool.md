# uv run tool

Run a Command Line Tool distributed as a Python package. Packages are
automatically download and installed into a cached, ephemeral, and
isolated environment on the first run.

## Usage

``` r
uv_run_tool(
  tool,
  args = character(),
  ...,
  from = NULL,
  with = NULL,
  python_version = NULL,
  exclude_newer = NULL
)
```

## Arguments

- tool, args:

  A character vector of command and arguments. Arguments are not quoted
  for the shell, so you may need to use
  [`shQuote()`](https://rdrr.io/r/base/shQuote.html).

- ...:

  Arguments passed on to
  [`base::system2`](https://rdrr.io/r/base/system2.html)

  `stdout,stderr`

  :   where output to `stdout` or `stderr` should be sent. Possible
      values are `""`, to the R console (the default), `NULL` or `FALSE`
      (discard output), `TRUE` (capture the output in a character
      vector) or a character string naming a file.

  `stdin`

  :   should input be diverted? `""` means the default, alternatively a
      character string naming a file. Ignored if `input` is supplied.

  `input`

  :   if a character vector is supplied, this is copied one string per
      line to a temporary file, and the standard input of `command` is
      redirected to the file.

  `env`

  :   character vector of name=value strings to set environment
      variables.

  `wait`

  :   a logical (not `NA`) indicating whether the R interpreter should
      wait for the command to finish, or run it asynchronously. This
      will be ignored (and the interpreter will always wait) if
      `stdout = TRUE` or `stderr = TRUE`. When running the command
      asynchronously, no output will be displayed on the `Rgui` console
      in Windows (it will be dropped, instead).

  `timeout`

  :   timeout in seconds, ignored if 0. This is a limit for the elapsed
      time running `command` in a separate process. Fractions of seconds
      are ignored.

  `receive.console.signals`

  :   a logical (not `NA`) indicating whether the command should receive
      events from the terminal/console that R runs from, particularly
      whether it should be interrupted by Ctrl-C. This will be ignored
      and events will always be received when `intern = TRUE` or
      `wait = TRUE`.

  `minimized,invisible`

  :   arguments that are accepted on Windows but ignored on this
      platform, with a warning.

- from:

  Use the given Python package to provide the command.

- with:

  Run with the given Python packages installed. You can also specify
  version constraints like `"ruff>=0.3.0"`.

- python_version:

  A Python version string, or character vector of Python version
  constraints.

- exclude_newer:

  String. Limit package versions to those published before a specified
  date. This offers a lightweight alternative to freezing package
  versions, helping guard against Python package updates that break a
  workflow. Accepts strings formatted as RFC 3339 timestamps (e.g.,
  `"2006-12-02T02:07:43Z"`) and local dates in the same format (e.g.,
  `"2006-12-02"`) in your system's configured time zone.

## Value

Return value of [`system2()`](https://rdrr.io/r/base/system2.html)

## Details

### Examples

    uv_run_tool("pycowsay", shQuote("hello from reticulate"))
    uv_run_tool("markitdown", shQuote(file.path(R.home("doc"), "NEWS.pdf")), stdout = TRUE)
    uv_run_tool("kaggle competitions download -c dogs-vs-cats")
    uv_run_tool("ruff", "--help")
    uv_run_tool("ruff format", shQuote(Sys.glob("**.py")))
    uv_run_tool("http", from = "httpie")
    uv_run_tool("http", "--version", from = "httpie<3.2.4", stdout = TRUE)
    uv_run_tool("saved_model_cli", "--help", from = "tensorflow")

## See also

<https://docs.astral.sh/uv/guides/tools/>
