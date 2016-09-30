
#' Parse script FLAGS from command line
#'
#' Parse command line arguments to the Rscript and use them to
#' populate the values of TensorFlow FLAGS
#'
#' @export
parse_flags <- function() {

  # get R command line arguments
  args <- commandArgs(trailingOnly = TRUE)

  # parse known arguments using the global parser
  parser <- tf$python$platform$flags$`_global_parser`
  result <- tryCatch(parser$parse_known_args(list(args)),
                     error = function(e) NULL)

  # check for error (means user invoked --help)
  if (is.null(result))
    quit(save = "no")

  # set flags from result
  FLAGS <- tf$app$flags$FLAGS
  result <- result[[1]]$`__dict__`
  for (name in names(result))
    FLAGS$`__setattr__`(name, result[[name]])

  invisible(NULL)
}
