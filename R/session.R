
#' @export
session <- function(target = "", graph = NULL, config = NULL) {
  tf <- tf_import()
  tf$Session(target, graph, config)
}


#' @export
interactive_session <- function(target = "", graph = NULL, config = NULL) {
  tf <- tf_import()
  tf$InteractiveSession(target, graph, config)
}

#' @export
run <- function(session, fetches, feed_dict=NULL,
                options=NULL, run_metadata=NULL) {
  tf <- tf_import()
  session$run(fetches, feed_dict, options, run_metadata)
}

