

#' @export
tf.Session <- function(target = "", graph = NULL, config = NULL) {
  session <- tf$Session(target, graph, config)
  reg.finalizer(session, function(sess) { sess$close() }, onexit = TRUE)
  tf.SessionClass$new(session = session)
}

#' @export
tf.InteractiveSession <- function(target = "", graph = NULL, config = NULL) {
  session <- tf$InteractiveSession(target, graph, config)
  reg.finalizer(session, function(sess) { sess$close() }, onexit = TRUE)
  tf.SessionClass$new(session = session)
}

#' @export
tf.initialize_all_variables <- function() {
  tf$initialize_all_variables()
}

tf.SessionClass <- setRefClass(
  Class = "tf.SessionClass",
  fields = c("session"),
  methods = list(
    run = function(fetches, feed_dict=NULL, options=NULL, run_metadata=NULL) {
      session$run(fetches, feed_dict, options, run_metadata)
    }
  )
)
