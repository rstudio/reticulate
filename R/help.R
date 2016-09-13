# RStudio IDE custom help handlers

help_handler <- function(type = c("completion", "parameter", "url"), topic, source, ...) {
  type <- match.arg(type)
  if (type == "completion") {
    help_completion_handler.tensorflow.python.object(topic, source)
  } else if (type == "parameter") {
    help_completion_parameter_handler.tensorflow.python.object(source)
  } else if (type == "url") {
    help_url_handler.tensorflow.python.object(topic, source)
  }
}

help_completion_handler.tensorflow.python.object <- function(topic, source) {

  # convert source to object if necessary
  source <- source_as_object(source)
  if (is.null(source))
    return(NULL)

  # use the first paragraph of the docstring as the description
  inspect <- import("inspect")
  description <- inspect$getdoc(py_get_attr(source, topic))
  if (is.null(description))
    description <- ""
  matches <- regexpr(pattern ='\n', description, fixed=TRUE)
  if (matches[[1]] != -1)
    description <- substring(description, 1, matches[[1]])
  description <- convert_description_types(description)

  # try to generate a signature
  signature <- NULL
  target <- py_get_attr(source, topic)
  if (py_is_callable(target)) {
    help <- import("tftools.help")
    signature <- help$generate_signature_for_function(target)
    if (is.null(signature))
      signature <- "()"
    signature <- paste0(topic, signature)
  }

  list(title = topic,
       signature = signature,
       description = description)
}

help_completion_parameter_handler.tensorflow.python.object <- function(source) {

  # split into topic and source
  components <- source_components(source)
  if (is.null(components))
    return(NULL)
  topic <- components$topic
  source <- components$source

  # get the function
  target <- py_get_attr(source, topic)
  if (py_is_callable(target)) {
    help <- import("tftools.help")
    args <- help$get_arguments(target)
    if (!is.null(args)) {
      # get the descriptions
      doc <- help$get_doc(target)
      if (is.null(doc))
        arg_descriptions <- args
      else {
        doc <- strsplit(doc, "\n", fixed = TRUE)[[1]]
        arg_descriptions <- sapply(args, function(arg) {
          prefix <- paste0("  ", arg, ": ")
          arg_line <- which(grepl(paste0("^", prefix), doc))
          if (length(arg_line) > 0) {
            arg_description <- substring(doc[[arg_line]], nchar(prefix))
            next_line <- arg_line + 1
            while((arg_line + 1) <= length(doc)) {
              line <- doc[[arg_line + 1]]
              if (grepl("^    ", line)) {
                arg_description <- paste(arg_description, line)
                arg_line <- arg_line + 1
              }
              else
                break
            }
            arg_description <- convert_description_types(arg_description)
          } else {
            arg
          }
        })
      }
      return(list(
        args = args,
        arg_descriptions = arg_descriptions
      ))
    }
  }

  NULL
}



help_url_handler.tensorflow.python.object <- function(topic, source) {

  # normalize topic and source for various calling scenarios
  if (grepl(" = $", topic)) {
    components <- source_components(source)
    if (is.null(components))
      return(NULL)
    topic <- components$topic
    source <- components$source
  } else {
    source <- source_as_object(source)
    if (is.null(source))
      return(NULL)
  }

  # get help page
  page <- NULL
  inspect <- import("inspect")
  if (inspect$ismodule(source)) {
    module <- paste(source$`__name__`)
    help <- module_help(module, topic)
  } else {
    help <- class_help(class(source), topic)
  }

  if (nzchar(help)) {
    version <- tf$`__version__`
    version <- strsplit(version, ".", fixed = TRUE)[[1]]
    help <- paste0("https://www.tensorflow.org/versions/r",
                   version[1], ".", version[2], "/api_docs/python/",
                   help)
  }

  # return help (can be "")
  help
}


help_formals_handler.tensorflow.python.object <- function(topic, source) {

  target <- py_get_attr(source, topic)
  if (py_is_callable(target)) {
    help <- import("tftools.help")
    args <- help$get_arguments(target)
    if (!is.null(args)) {
      return(list(
        formals = args,
        helpHandler = "tensorflow:::help_handler"
      ))
    }
  }

  # default to NULL if we couldn't get the arguments
  NULL
}

# convert types in description
convert_description_types <- function(description) {
  description <- sub("`None`", "`NULL`", description)
  description <- sub("`True`", "`TRUE`", description)
  description <- sub("`False`", "`FALSE`", description)
  description
}

# convert source to object if necessary
source_as_object <- function(source) {

  if (is.character(source)) {
    source <- tryCatch(eval(parse(text = source), envir = globalenv()),
                       error = function(e) NULL)
    if (is.null(source))
      return(NULL)
  }

  source
}

source_components <- function(source) {
  components <- strsplit(source, "\\$")[[1]]
  topic <- components[[length(components)]]
  source <- paste(components[1:(length(components)-1)], collapse = "$")
  source <- source_as_object(source)
  if (!is.null(source))
    list(topic = topic, source = source)
  else
    NULL
}


module_help <- function(module, topic) {

  # do we have a page for this module/topic?
  lookup <- paste(module, topic, sep = ".")
  page <- .module_help_pages[[lookup]]

  # if so then append topic
  if (!is.null(page))
    paste(page, topic, sep = "#")
  else
    ""
}

class_help <- function(class, topic) {

  # call recursively for more than one class
  if (length(class) > 1) {
    # call for each class
    for (i in 1:length(class)) {
      help <- class_help(class[[i]], topic)
      if (nzchar(help))
        return(help)
    }
    # no help found
    return("")
  }

  # do we have a page for this class?
  page <- .class_help_pages[[class]]

  # if so then append class and topic
  if (!is.null(page)) {
    components <- strsplit(class, ".", fixed = TRUE)[[1]]
    class <- components[[length(components)]]
    paste0(page, "#", class, ".", topic)
  } else {
    ""
  }
}

help_page <- function(page, prefix, topics) {
  names <- paste(prefix, topics, sep = ".")
  values <- rep_len(page, length(names))
  names(values) <- names
  values
}

help_pages <- function(...) {
  pages <- c(...)
  list2env(parent = emptyenv(), as.list(pages))
}


.module_help_pages <- help_pages(
  help_page("framework.html", "tensorflow", c(
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
    "bytes")
  ),
  help_page("constant_op.html", "tensorflow", c(
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
  )),
  help_page("state_ops.html", "tensorflow", c(
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
  )),
  help_page("state_ops.html", "tensorflow.python.training.training", c(
    "Saver",
    "latest_checkpoint",
    "get_checkpoint_state",
    "update_checkpoint_state"
  )),
  help_page("array_ops.html", "tensorflow", c(
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
  )),
  help_page("math_ops.html", "tensorflow", c(
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
  )),
  help_page("control_flow_ops.html", "tensorflow", c(
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
  )),
  help_page("image.html", "tensorflow.python.ops.image_ops", c(
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
  )),
  help_page("sparse_ops.html", "tensorflow", c(
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
  )),
  help_page("io_ops.html", "tensorflow", c(
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
  )),
  help_page("io_ops.html", "tensorflow.python.training.training", c(
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
  ))
)

# Next up is Data IO (Python functions): https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#data-io-python-functions


.class_help_pages <- help_pages(
  help_page("framework.html", "tensorflow.python.framework", c(
    "ops.Graph",
    "ops.Operation",
    "ops.Tensor",
    "dtypes.DType",
    "ops.GraphKeys",
    "ops.RegisterGradient",
    "tensor_shape.TensorShape",
    "tensor_shape.Dimension",
    "device.DeviceSpec"
  )),
  help_page("state_ops.html", "tensorflow.python", c(
    "ops.variables.Variable",
    "training.saver.Saver",
    "ops.variable_scope.VariableScope",
    "framework.ops.IndexedSlices"
  )),
  help_page("sparse_ops.html", "tensorflow.python.framework.ops", c(
    "SparseTensor",
    "SparseTensorValue"
  )),
  help_page("io_ops.html", "tensorflow.python.ops.io_ops", c(
    "BaseReader",
    "TextLineReader",
    "WholeFileReader",
    "IdentityReader",
    "TFRecordReader",
    "FixedLengthRecordReader"
  )),
  help_page("io_ops.html", "tensorflow.python.ops.data_flow_ops", c(
    "QueueBase"
  ))
)


