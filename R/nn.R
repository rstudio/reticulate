
#' @export
tf.nn.softmax <- function(logits, name=NULL) {
  tf$nn$softmax(logits, name)
}
