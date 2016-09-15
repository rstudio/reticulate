
#' Main TensorFlow module
#'
#' Interface to main TensorFlow  module. Provides access to top level classes
#' and functions as well as sub-modules (e.g. \code{tf$nn},
#' \code{tf$contrib$learn}, etc.).
#'
#' @format TensorFlow module
#'
#' @examples
#' \dontrun{
#' hello <- tf$constant('Hello, TensorFlow!')
#' b <- tf$Variable(tf$zeros(list(1L)))
#'
#' sess <- tf$Session()
#' sess$initialize_all_variables()
#'
#' learn <- tf$contrib$learn
#' slim <- tf$contrib$slim
#' }
#' @export
tf <- NULL
