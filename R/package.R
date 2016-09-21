

# TODO: NaN coming back for loss when running fully_connected_feed.R on py3

# TODO: periodic error on Linux in fully_connected_feed.R (py2 and py3):
# 1. Failure: mnist/fully_connected_feed.R example runs successfully (@test-examples.R#30)
  # run_example(example) threw an error.
  # InvalidArgumentError: targets[98] is out of range
  # [[Node: InTopK = InTopK[T=DT_INT32, k=1, _device="/job:localhost/replica:0/task:0/cpu:0"](softmax_linear/Add, _recv_Placeholder_1_0)]]
  # Caused by op u'InTopK', defined at:
  #   File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_nn_ops.py", line 973, in in_top_k
  # targets=targets, k=k, name=name)
  # File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 703, in apply_op
  # op_def=op_def)
  # File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 2310, in create_op
  # original_op=self._default_original_op, op_def=op_def)
  # File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1232, in __init__
  # self._traceback = _extract_stack()

# TODO: add TENSORFLOW_PYTHON environment variable

# TODO: completion for np$absolute causes an error

# TODO: add docs on TENSORFLOW_PYTHON_VERSION python3
# TODO: revise modules section in api doc
# TODO: port selected "how to" docs
# TODO: port additional tutorials


#' @useDynLib tensorflow
#' @importFrom Rcpp evalCpp
NULL

.onLoad <- function(libname, pkgname) {

  # initialize python
  config <- py_config()
  py_initialize(config$libpython);

  # add our python scripts to the search path
  py_run_string(paste0("import sys; sys.path.append('",
                       system.file("python", package = "tensorflow") ,
                       "')"))

  # call tf onLoad handler
  tf_on_load(libname, pkgname)
}


.onAttach <- function(libname, pkgname) {
  # call tf onAttach handler
  tf_on_attach(libname, pkgname)
}

.onUnload <- function(libpath) {
  py_finalize();
}
