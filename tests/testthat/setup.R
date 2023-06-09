
## CRAN reports we create a
# if (!exists("~/.virtualenvs"))
#   withr::defer(unlink("~/.virtualenvs", recursive = TRUE), teardown_env())


.oop <- list(
  Matrix.warnDeprecatedCoerce = 2L,
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE
)

if(getRversion() <= "3.5") {
  .oop$warnPartialMatchArgs <- NULL
}

.oop <- do.call(options, .oop)
withr::defer(options(.oop), teardown_env())
