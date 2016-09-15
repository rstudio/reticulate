
# Register all help topics
register_tf_help_topics <- function() {

  # get the tensorflow help url
  version <- tf$`__version__`
  version <- strsplit(version, ".", fixed = TRUE)[[1]]
  help_url <- paste0("https://www.tensorflow.org/versions/r",
                     version[1], ".", version[2], "/api_docs/python/")

  # helper function to make tf help topics
  tf_help_topics <- function(page, prefix, symbols) {
    help_topics(paste0(help_url, page), prefix, symbols)
  }

  # register topics
  register_help_topics("module", tf_help_topics("framework.html", "tensorflow", c(
    "Graph",
    "Operation",
    "Tensor",
    "DType",
    "as_dtype",
    "device",
    "container",
    "name_scope",
    "control_dependencies",
    "convert_to_tensor",
    "convert_to_tensor_or_indexed_slices",
    "get_default_graph",
    "reset_default_graph",
    "import_graph_def",
    "load_file_system_library",
    "load_op_library",
    "add_to_collection",
    "get_collection",
    "get_collection_ref",
    "GraphKeys",
    "RegisterGradient",
    "NoGradient",
    "RegisterShape",
    "TensorShape",
    "Dimension",
    "op_scope",
    "register_tensor_conversion_function",
    "DeviceSpec",
    "bytes"
  )))


  register_help_topics("module", tf_help_topics("constant_op.html", "tensorflow", c(
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "fill",
    "constant",
    "range",
    "random_normal",
    "truncated_normal",
    "random_uniform",
    "random_shuffle ",
    "random_crop",
    "multinomial",
    "random_gamma",
    "set_random_seed",
    "contrib.graph_editor.ops"
  )))

  register_help_topics("module", tf_help_topics("state_ops.html", "tensorflow", c(
    "Variable",
    "all_variables",
    "trainable_variables",
    "local_variables",
    "moving_average_variables",
    "initialize_all_variables",
    "initialize_local_variables",
    "is_variable_initialized",
    "report_uninitialized_variables",
    "assert_variables_initialized",
    "get_variable",
    "VariableScope",
    "variable_scope",
    "variable_op_scope",
    "get_variable_scope",
    "make_template",
    "no_regularizer",
    "constant_initializer",
    "random_normal_initializer",
    "truncated_normal_initializer",
    "random_uniform_initializer",
    "uniform_unit_scaling_initializer",
    "zeros_initializer",
    "ones_initializer",
    "variable_axis_size_partitioner",
    "min_max_variable_partitioner",
    "scatter_update",
    "scatter_add",
    "scatter_sub",
    "sparse_mask",
    "IndexedSlices",
    "export_meta_graph",
    "import_meta_graph"
  )))

  register_help_topics("module", tf_help_topics("state_ops.html", "tensorflow.python.training.training", c(
    "Saver",
    "latest_checkpoint",
    "get_checkpoint_state",
    "update_checkpoint_state"
  )))

  register_help_topics("module", tf_help_topics("array_ops.html", "tensorflow", c(
    "string_to_number",
    "to_double",
    "to_float",
    "to_bfloat16",
    "to_int32",
    "to_int64",
    "cast",
    "saturate_cast",
    "shape",
    "size",
    "rank",
    "reshape",
    "squeeze",
    "expand_dims",
    "meshgrid",
    "slice",
    "strided_slice",
    "split",
    "tile",
    "pad",
    "concat",
    "pack",
    "unpack",
    "reverse_sequence",
    "reverse",
    "transpose",
    "extract_image_patches",
    "space_to_batch",
    "batch_to_space",
    "space_to_depth",
    "depth_to_space",
    "gather",
    "gather_nd",
    "dynamic_partition",
    "dynamic_stitch",
    "boolean_mask",
    "one_hot",
    "bitcast",
    "contrib.graph_editor.copy",
    "shape_n",
    "unique_with_counts"
  )))

  register_help_topics("module", tf_help_topics("math_ops.html", "tensorflow", c(
    "add",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "mod",
    "cross",
    "add_n",
    "abs",
    "neg",
    "sign",
    "inv",
    "square",
    "round",
    "sqrt",
    "rsqrt",
    "pow",
    "exp",
    "log",
    "ceil",
    "floor",
    "maximum",
    "minimum",
    "cos",
    "sin",
    "lbeta",
    "tan",
    "acos",
    "asin",
    "atan",
    "lgamma",
    "digamma",
    "erf",
    "erfc",
    "squared_difference",
    "igamma",
    "igammac",
    "zeta",
    "polygamma",
    "batch_matrix_diag",
    "batch_matrix_diag_part",
    "batch_matrix_set_diag",
    "diag",
    "diag_part",
    "trace",
    "transpose",
    "batch_matrix_transpose",
    "matmul",
    "batch_matmul",
    "matrix_determinant",
    "batch_matrix_determinant",
    "matrix_inverse",
    "batch_matrix_inverse",
    "cholesky",
    "batch_cholesky",
    "cholesky_solve",
    "batch_cholesky_solve",
    "matrix_solve",
    "batch_matrix_solve",
    "matrix_triangular_solve",
    "batch_matrix_triangular_solve",
    "matrix_solve_ls",
    "batch_matrix_solve_ls",
    "self_adjoint_eig",
    "batch_self_adjoint_eig",
    "self_adjoint_eigvals",
    "batch_self_adjoint_eigvals",
    "svd",
    "batch_svd",
    "complex",
    "complex_abs",
    "conj",
    "imag",
    "real",
    "fft",
    "ifft",
    "fft2d",
    "ifft2d",
    "fft3d",
    "ifft3d",
    "batch_fft",
    "batch_ifft",
    "batch_fft2d",
    "batch_ifft2d",
    "batch_fft3d",
    "batch_ifft3d",
    "reduce_sum",
    "reduce_prod",
    "reduce_min",
    "reduce_max",
    "reduce_mean",
    "reduce_all",
    "reduce_any",
    "accumulate_n",
    "cumsum",
    "cumprod",
    "segment_sum",
    "segment_prod",
    "segment_min",
    "segment_max",
    "segment_mean",
    "unsorted_segment_sum",
    "sparse_segment_sum",
    "sparse_segment_mean",
    "sparse_segment_sqrt_n",
    "argmin",
    "argmax",
    "listdiff",
    "where",
    "unique",
    "edit_distance",
    "invert_permutation",
    "scalar_mul",
    "sparse_segment_sqrt_n_grad"
  )))

  register_help_topics("module", tf_help_topics("control_flow_ops.html", "tensorflow", c(
    "identity",
    "tuple",
    "group",
    "no_op",
    "count_up_to",
    "cond",
    "case",
    "while_loop",
    "logical_and",
    "logical_or",
    "logical_xor",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "select",
    "where",
    "is_finite",
    "is_inf",
    "is_nan",
    "verify_tensor_all_finite",
    "check_numerics",
    "add_check_numerics_ops",
    "Assert",
    "Print"
  )))

  register_help_topics("module", tf_help_topics("image.html", "tensorflow.python.ops.image_ops", c(
    "decode_jpeg",
    "encode_jpeg",
    "decode_png",
    "encode_png",
    "resize_images",
    "resize_area",
    "resize_bicubic",
    "resize_bilinear",
    "resize_nearest_neighbor",
    "resize_image_with_crop_or_pad",
    "central_crop",
    "pad_to_bounding_box",
    "crop_to_bounding_box",
    "extract_glimpse",
    "crop_and_resize",
    "flip_up_down",
    "random_flip_up_down",
    "flip_left_right",
    "random_flip_left_right",
    "transpose_image",
    "rot90",
    "rgb_to_grayscale",
    "grayscale_to_rgb",
    "hsv_to_rgb",
    "rgb_to_hsv",
    "convert_image_dtype",
    "adjust_brightness",
    "random_brightness",
    "adjust_contrast",
    "random_contrast",
    "adjust_hue",
    "random_hue",
    "adjust_saturation",
    "random_saturation",
    "per_image_whitening",
    "draw_bounding_boxes",
    "non_max_suppression",
    "sample_distorted_bounding_box"
  )))

  register_help_topics("module", tf_help_topics("sparse_ops.html", "tensorflow", c(
    "SparseTensor",
    "SparseTensorValue",
    "sparse_to_dense",
    "sparse_tensor_to_dense",
    "sparse_to_indicator",
    "sparse_merge",
    "sparse_concat",
    "sparse_reorder",
    "sparse_reshape",
    "sparse_split",
    "sparse_retain",
    "sparse_reset_shape",
    "sparse_fill_empty_rows",
    "sparse_reduce_sum",
    "sparse_add",
    "sparse_softmax",
    "sparse_tensor_dense_matmul",
    "sparse_maximum",
    "sparse_minimum"
  )))

  register_help_topics("module", tf_help_topics("io_ops.html", "tensorflow", c(
    "placeholder",
    "placeholder_with_default",
    "sparse_placeholder",
    "BaseReader",
    "TextLineReader",
    "WholeFileReader",
    "IdentityReader",
    "TFRecordReader",
    "FixedLengthRecordReader",
    "decode_csv",
    "decode_raw",
    "VarLenFeature",
    "FixedLenFeature",
    "FixedLenSequenceFeature",
    "parse_example",
    "parse_single_example",
    "decode_json_example",
    "QueueBase",
    "FIFOQueue",
    "PaddingFIFOQueue",
    "RandomShuffleQueue",
    "matching_files",
    "read_file"
  )))

  register_help_topics("module", tf_help_topics("io_ops.html", "tensorflow.python.training.training", c(
    "match_filenames_once",
    "limit_epochs",
    "input_producer",
    "range_input_producer",
    "slice_input_producer",
    "string_input_producer",
    "batch",
    "batch_join",
    "shuffle_batch",
    "shuffle_batch_join"
  )))

  register_help_topics("module", tf_help_topics("python_io.html", "tensorflow.python.lib.io.python_io", c(
    "TFRecordWriter",
    "tf_record_iterator"
  )))

  register_help_topics("module", tf_help_topics("nn.html", "tensorflow.python.ops.nn", c(
    "relu",
    "relu6",
    "elu",
    "softplus",
    "softsign",
    "dropout",
    "bias_add",
    "sigmoid",
    "tanh",
    "conv2d",
    "depthwise_conv2d",
    "separable_conv2d",
    "atrous_conv2d",
    "conv2d_transpose",
    "conv3d",
    "avg_pool",
    "max_pool",
    "max_pool_with_argmax",
    "avg_pool3d",
    "max_pool3d",
    "dilation2d",
    "erosion2d",
    "l2_normalize",
    "local_response_normalization",
    "sufficient_statistics",
    "normalize_moments",
    "moments",
    "l2_loss",
    "log_poisson_loss",
    "sigmoid_cross_entropy_with_logits",
    "softmax",
    "log_softmax",
    "softmax_cross_entropy_with_logits",
    "sparse_softmax_cross_entropy_with_logits",
    "weighted_cross_entropy_with_logits",
    "embedding_lookup",
    "embedding_lookup_sparse",
    "dynamic_rnn",
    "rnn",
    "state_saving_rnn",
    "bidirectional_rnn",
    "ctc_loss",
    "ctc_greedy_decoder",
    "ctc_beam_search_decoder",
    "top_k",
    "in_top_k",
    "nce_loss",
    "sampled_softmax_loss",
    "uniform_candidate_sampler",
    "log_uniform_candidate_sampler",
    "learned_unigram_candidate_sampler",
    "fixed_unigram_candidate_sampler",
    "compute_accidental_hits",
    "batch_normalization",
    "depthwise_conv2d_native"
  )))

  register_help_topics("module",  tf_help_topics("client.html", "tensorflow", c(
    "Session",
    "InteractiveSession",
    "get_default_session",
    "OpError"
  )))

  register_help_topics("module", tf_help_topics("client.html", "tensorflow.python.framework.errors", c(
    "OpError",
    "CancelledError",
    "UnknownError",
    "InvalidArgumentError",
    "DeadlineExceededError",
    "NotFoundError",
    "AlreadyExistsError",
    "PermissionDeniedError",
    "UnauthenticatedError",
    "ResourceExhaustedError",
    "FailedPreconditionError",
    "AbortedError",
    "OutOfRangeError",
    "UnimplementedError",
    "InternalError",
    "UnavailableError",
    "DataLossError"
  )))

  register_help_topics("module", tf_help_topics("train.html", "tensorflow.python.training.training", c(
    "Optimizer",
    "GradientDescentOptimizer",
    "AdadeltaOptimizer",
    "AdagradOptimizer",
    "MomentumOptimizer",
    "AdamOptimizer",
    "FtrlOptimizer",
    "RMSPropOptimizer",
    "exponential_decay",
    "ExponentialMovingAverage",
    "Coordinator",
    "QueueRunner",
    "add_queue_runner",
    "start_queue_runners",
    "Server",
    "Supervisor",
    "SessionManager",
    "ClusterSpec",
    "replica_device_setter",
    "SummaryWriter",
    "summary_iterator",
    "global_step",
    "write_graph",
    "LooperThread",
    "do_quantize_training_on_graphdef",
    "generate_checkpoint_state_proto"
  )))

  register_help_topics("module", tf_help_topics("train.html", "tensorflow", c(
    "gradients",
    "AggregationMethod",
    "stop_gradient",
    "clip_by_value",
    "clip_by_norm",
    "clip_by_average_norm",
    "clip_by_global_norm",
    "global_norm",
    "scalar_summary",
    "image_summary",
    "audio_summary",
    "histogram_summary",
    "merge_summary",
    "merge_all_summaries"
  )))

  register_help_topics("module", tf_help_topics("train.html", "tensorflow.python.ops.nn", c(
    "zero_fraction"
  )))

  register_help_topics("module", tf_help_topics("script_ops.html", "tensorflow", c(
    "py_func"
  )))

  register_help_topics("module", tf_help_topics("test.html", "tensorflow.python.platform.test", c(
    "main",
    "assert_equal_graph_def",
    "get_temp_dir",
    "is_built_with_cuda",
    "compute_gradient",
    "compute_gradient_error"
  )))

  register_help_topics("module", tf_help_topics("contrib.layers.html", "tensorflow.contrib.layers", c(
    "avg_pool2d",
    "batch_norm",
    "convolution2d",
    "convolution2d_in_plane",
    "convolution2d_transpose",
    "flatten",
    "fully_connected",
    "max_pool2d",
    "one_hot_encoding",
    "repeat",
    "separable_convolution2d",
    "stack",
    "unit_norm",
    "apply_regularization",
    "l1_regularizer",
    "l2_regularizer",
    "sum_regularizer",
    "xavier_initializer",
    "xavier_initializer_conv2d",
    "variance_scaling_initializer",
    "optimize_loss",
    "summarize_activation",
    "summarize_tensor",
    "summarize_tensors",
    "summarize_collection",
    "summarize_activations"
  )))

  register_help_topics("module", tf_help_topics("contrib.losses.html", "tensorflow.contrib.losses", c(
    "absolute_difference",
    "add_loss",
    "cosine_distance",
    "get_losses",
    "get_regularization_losses",
    "get_total_loss",
    "hinge_loss",
    "log_loss",
    "sigmoid_cross_entropy",
    "softmax_cross_entropy",
    "sum_of_pairwise_squares",
    "sum_of_squares"
  )))

  register_help_topics("module", tf_help_topics("contrib.metrics.html", "tensorflow.contrib.metrics", c(
    "streaming_accuracy",
    "streaming_mean",
    "streaming_recall",
    "streaming_precision",
    "streaming_auc",
    "streaming_recall_at_k",
    "streaming_mean_absolute_error",
    "streaming_mean_iou",
    "streaming_mean_relative_error",
    "streaming_mean_squared_error",
    "streaming_root_mean_squared_error",
    "streaming_mean_cosine_distance",
    "streaming_percentage_less",
    "streaming_sparse_precision_at_k",
    "streaming_sparse_recall_at_k",
    "auc_using_histogram",
    "accuracy",
    "confusion_matrix",
    "aggregate_metrics",
    "aggregate_metric_map",
    "set_difference",
    "set_intersection",
    "set_size",
    "set_union"
  )))

  register_help_topics("module", tf_help_topics("contrib.learn.html", "tensorflow.contrib.learn", c(
    "BaseEstimator",
    "Estimator",
    "ModeKeys",
    "TensorFlowClassifier",
    "DNNClassifier",
    "DNNRegressor",
    "TensorFlowDNNClassifier",
    "TensorFlowDNNRegressor",
    "TensorFlowEstimator",
    "LinearClassifier",
    "LinearRegressor",
    "TensorFlowLinearClassifier",
    "TensorFlowLinearRegressor",
    "TensorFlowRNNClassifier",
    "TensorFlowRNNRegressor",
    "TensorFlowRegressor",
    "NanLossDuringTrainingError",
    "RunConfig",
    "evaluate",
    "infer",
    "run_feeds",
    "run_n",
    "train",
    "extract_dask_data",
    "extract_dask_labels",
    "extract_pandas_data",
    "extract_pandas_labels",
    "extract_pandas_matrix",
    "read_batch_examples",
    "read_batch_features",
    "read_batch_record_features"
  )))

  register_help_topics("module", tf_help_topics("contrib.framework.html", "tensorflow.contrib.framework", c(
    "assert_same_float_dtype",
    "assert_scalar_int",
    "convert_to_tensor_or_sparse_tensor",
    "get_graph_from_inputs",
    "is_tensor",
    "reduce_sum_n",
    "safe_embedding_lookup_sparse",
    "with_shape",
    "with_same_shape",
    "deprecated",
    "deprecated_arg_values",
    "arg_scope",
    "add_arg_scope",
    "has_arg_scope",
    "arg_scoped_arguments",
    "add_model_variable",
    "assert_global_step",
    "assert_or_get_global_step",
    "create_global_step",
    "get_global_step",
    "get_or_create_global_step",
    "get_local_variables",
    "get_model_variables",
    "get_unique_variable",
    "get_variables_by_name",
    "get_variables_by_suffix",
    "get_variables_to_restore",
    "get_variables",
    "local_variable",
    "model_variable",
    "variable",
    "VariableDeviceChooser"
  )))

  register_help_topics("module", tf_help_topics("contrib.framework.html", "tensorflow", c(
    "is_numeric_tensor",
    "is_non_decreasing",
    "is_strictly_increasing"
  )))

  register_help_topics("module", tf_help_topics("contrib.util.html", "tensorflow.contrib.util", c(
    "constant_value",
    "make_tensor_proto",
    "make_ndarray",
    "ops_used_by_graph_def",
    "stripped_op_list_for_graph"
  )))

  register_help_topics("class", tf_help_topics("framework.html", "tensorflow.python.framework", c(
    "ops.Graph",
    "ops.Operation",
    "ops.Tensor",
    "dtypes.DType",
    "ops.GraphKeys",
    "ops.RegisterGradient",
    "tensor_shape.TensorShape",
    "tensor_shape.Dimension",
    "device.DeviceSpec"
  )))

  register_help_topics("class", tf_help_topics("state_ops.html", "tensorflow.python", c(
    "ops.variables.Variable",
    "training.saver.Saver",
    "ops.variable_scope.VariableScope",
    "framework.ops.IndexedSlices"
  )))

  register_help_topics("class", tf_help_topics("sparse_ops.html", "tensorflow.python.framework.ops", c(
    "SparseTensor",
    "SparseTensorValue"
  )))

  register_help_topics("class", tf_help_topics("io_ops.html", "tensorflow.python.ops.io_ops", c(
    "BaseReader",
    "TextLineReader",
    "WholeFileReader",
    "IdentityReader",
    "TFRecordReader",
    "FixedLengthRecordReader"
  )))

  register_help_topics("class", tf_help_topics("io_ops.html", "tensorflow.python.ops.data_flow_ops", c(
    "QueueBase"
  )))

  register_help_topics("class", tf_help_topics("python_io.html", "tensorflow.python.lib.io", c(
    "tf_record.TFRecordWriter"
  )))

  register_help_topics("class", tf_help_topics("client.html", "tensorflow.python.client.session", c(
    "Session"
  )))

  register_help_topics("class", tf_help_topics("client.html", "tensorflow.python.framework.errors", c(
    "OpError"
  )))

  register_help_topics("class", tf_help_topics("train.html", "tensorflow.python.training", c(
    "optimizer.Optimizer",
    "moving_averages.ExponentialMovingAverage",
    "coordinator.Coordinator",
    "queue_runner.QueueRunner",
    "server_lib.Server",
    "supervisor.Supervisor",
    "session_manager.SessionManager",
    "server_lib.ClusterSpec",
    "summary_io.SummaryWriter",
    "coordinator.LooperThread"
  )))

  register_help_topics("class", tf_help_topics("contrib.learn.html", "tensorflow.contrib.learn.python.learn.estimators", c(
    "BaseEstimator",
    "Estimator",
    "ModeKeys",
    "dnn.DNNClassifier",
    "dnn.DNNRegressor",
    "linear.LinearClassifier",
    "linear.LinearRegressor",
    "run_config.RunConfig"
  )))
}
