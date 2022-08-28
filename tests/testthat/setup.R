
## CRAN reports we create a
# if(!exists("~/.virtualenvs"))
#   withr::defer(unlink("~/.virtualenvs", recursive = TRUE), teardown_env())


.oop <- options(
  Matrix.warnDeprecatedCoerce = 2L,
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE
)

withr::defer(options(.oop), teardown_env())
