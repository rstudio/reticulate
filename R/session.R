
#' Create a new TensorFlow session
#'
#' A Session object encapsulates the environment in which Operation objects are
#' executed, and Tensor objects are evaluated.
#'
#' @param target Optional. The execution engine to connect to. Defaults to using
#'   an in-process engine.
#' @param graph: Optional. The Graph to be launched (described below).
#' @param config (Optional.) A list with configuration options for the session.
#'
#' @details
#'
#' If no \code{graph} argument is specified when constructing the session, the
#' default graph will be launched in the session. If you are using more than one
#' graph (created with \code{graph} in the same process, you will have to use
#' different sessions for each graph, but each graph can be used in multiple
#' sessions. In this case, it is often clearer to pass the graph to be launched
#' explicitly to the session function
#'
#'
#' @export
session <- function(target = "", graph = NULL, config = NULL) {
  tf <- tf_import()
  sess <- tf$Session(target, graph, config)
  reg.finalizer(sess, function(sess) { sess$close() }, onexit = TRUE)
  sess
}


#' @export
interactive_session <- function(target = "", graph = NULL, config = NULL) {
  tf <- tf_import()
  tf$InteractiveSession(target, graph, config)
}

#' @export
run <- function(session, fetches, feed_dict=NULL,
                options=NULL, run_metadata=NULL) {
  session$run(fetches, feed_dict, options, run_metadata)
}

