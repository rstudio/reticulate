
.engine_context <- new.env(parent = emptyenv())

#' A reticulate Engine for Knitr
#'
#' This provides a `reticulate` engine for `knitr`, suitable for usage when
#' attempting to render Python chunks. Using this engine allows for shared state
#' between Python chunks in a document -- that is, variables defined by one
#' Python chunk can be used by later Python chunks.
#'
#' The engine can be activated by setting (for example)
#'
#' ```
#' knitr::knit_engines$set(python = reticulate::eng_python)
#' ```
#'
#' Typically, this will be set within a document's setup chunk, or by the
#' environment requesting that Python chunks be processed by this engine.
#' Note that `knitr` (since version 1.18) will use the `reticulate` engine by
#' default when executing Python chunks within an R Markdown document.
#'
#' @param options
#'   Chunk options, as provided by `knitr` during chunk execution.
#'
#' @section Supported `knitr` chunk options:
#'
#' For most options, reticulate's python engine behaves the same as the default
#' R engine included in knitr, but they might not support all the same features.
#' Options in *italic* are equivalent to knitr, but with modified behavior.
#'
#' - *`eval`* (`TRUE`, logical): If `TRUE`, all expressions in the chunk are evaluated. If `FALSE`,
#'   no expression is evaluated. Unlike knitr's R engine, it doesn't support numeric
#'   values indicating the expressions to evaluate.
#' - *`echo`* (`TRUE`, logical): Whether to display the source code in the output document. Unlike
#'   knitr's R engine, it doesn't support numeric values indicating the expressions
#'   to display.
#' - `results` (`'markup'`, character): Controls how to display the text results. Note that this option only
#'   applies to normal text output (not warnings, messages, or errors). The behavior
#'   should be identical to knitr's R engine.
#' - `collapse` (`FALSE`, logical): Whether to, if possible, collapse all the source and output blocks
#'   from one code chunk into a single block (by default, they are written to separate blocks).
#'   This option only applies to Markdown documents.
#' - `error` (`TRUE`, logical): Whether to preserve errors. If `FALSE` evaluation stops
#'   on errors. (Note that RMarkdown sets it to `FALSE`).
#' - *`warning`* (`TRUE`, logical): Whether to preserve warnings in the output. If FALSE, all warnings
#'   will be suppressed. Doesn't support indices.
#' - `include` (`TRUE`, logical): Whether to include the chunk output in the output document.
#'   If `FALSE`, nothing will be written into the output document, but the code is still
#'   evaluated and plot files are generated if there are any plots in the chunk, so you
#'   can manually insert figures later.
#' - `dev`: The graphical device to generate plot files. See knitr documentation for
#'    additional information.
#' - `base.dir` (`NULL`; character): An absolute directory under which the plots
#'    are generated.
#' - `strip.white` (TRUE; logical): Whether to remove blank lines in the beginning
#'   or end of a source code block in the output.
#' - `dpi` (72; numeric): The DPI (dots per inch) for bitmap devices (dpi * inches = pixels).
#' - `fig.width`, `fig.height` (both are 7; numeric): Width and height of the plot
#'   (in inches), to be used in the graphics device.
#' - `label`: The chunk label for each chunk is assumed to be unique within the
#'   document. This is especially important for cache and plot filenames, because
#'   these filenames are based on chunk labels. Chunks without labels will be
#'   assigned labels like unnamed-chunk-i, where i is an incremental number.
#'
#' ### Python engine only options
#'
#' - **`jupyter_compat`** (FALSE, logical): If `TRUE` then, like in Jupyter notebooks,
#'   only the last expression in the chunk is printed to the output.
#' - **`out.width.px`**, **`out.height.px`** (810, 400, both integers): Width and
#'   height of the plot in the output document, which can be different with its
#'   physical `fig.width` and `fig.height`, i.e., plots can be scaled in the output
#'   document. Unlike knitr's `out.width`, this is always set in pixels.
#' - **`altair.fig.width`**, **`altair.fig.height`**: If set, is used instead of
#'   `out.width.px` and `out.height.px` when writing Altair charts.
#'
#' @export
eng_python <- function(options) {

  # check for unsupported knitr options
  options <- eng_python_validate_options(options)

  # when 'eval = FALSE', we can just return the source code verbatim
  # (skip any other per-chunk work)
  if (identical(options$eval, FALSE)) {
    outputs <- list()
    if (!identical(options$echo, FALSE))
      outputs[[1]] <- structure(list(src = options$code), class = "source")
    wrap <- getOption("reticulate.engine.wrap", eng_python_wrap)
    return(wrap(outputs, options))
  }

  engine.path <- if (is.list(options[["engine.path"]]))
    options[["engine.path"]][["python"]]
  else
    options[["engine.path"]]

  # if the user has requested a custom Python, attempt
  # to honor that request (warn if Python already initialized
  # to a different version)
  if (is.character(engine.path)) {

    # if Python has not yet been loaded, then try
    # to load it with the requested version of Python
    if (!py_available())
      use_python(engine.path, required = TRUE)

    # double-check that we've loaded the requested Python
    conf <- py_config()
    requestedPython <- normalizePath(engine.path)
    actualPython <- normalizePath(conf$python)
    if (requestedPython != actualPython) {
      fmt <- "cannot honor request to use Python %s [%s already loaded]"
      msg <- sprintf(fmt, requestedPython, actualPython)
      warning(msg, immediate. = TRUE, call. = FALSE)
    }
  }

  # environment tracking the labels assigned to newly-created altair charts
  .engine_context$altair_ids <- new.env(parent = emptyenv())

  # a list of pending plots / outputs
  .engine_context$pending_plots <- stack()

  eng_python_initialize(options = options, envir = environment())

  # helper function for extracting range of code, dropping blank lines
  extract <- function(code, range) {
    snippet <- code[range[1]:range[2]]
    paste(snippet, collapse = "\n")
  }

  # extract the code to be run -- we'll attempt to run the code line by line
  # and detect changes so that we can interleave code and output (similar to
  # what one sees when executing an R chunk in knitr). to wit, we'll do our
  # best to emulate the return format of 'evaluate::evaluate()'
  code <- options$code
  n <- length(code)
  if (n == 0)
    return(list())

  # use 'ast.parse()' to parse Python code and collect line numbers, so we
  # can split source code into statements
  ast <- import("ast", convert = TRUE)
  pasted <- paste(code, collapse = "\n")
  parsed <- tryCatch(ast$parse(pasted, "<string>"), error = identity)
  if (inherits(parsed, "error")) {
    error <- reticulate::py_last_error()
    if (identical(options$error, TRUE)) {
      outputs <- list(
        structure(list(src = code), class = "source"),
        paste(error$value, collapse = "\n")
      )
      wrap <- getOption("reticulate.engine.wrap", eng_python_wrap)
      return(wrap(outputs, options))
    } else {
      stop(error$value, call. = FALSE)
    }
  }

  # iterate over top-level nodes and extract line numbers
  lines <- vapply(parsed$body, function(node) {
    if(py_version() >= "3.8")
      return(as_r_value(py_get_attr(node, "end_lineno")))
    # `end_lineno` attribute was introduced in python3.8
    # in earlier versions, fallback to using just lineno
    # note, this can result in comments being attached to
    # the wrong code chunk

    if (py_has_attr(node, "decorator_list") && length(node$decorator_list)) {
      out <- py_get_attr(node$decorator_list[[1]], "lineno")
    } else {
      out <- py_get_attr(node, "lineno")
    }

    as_r_value(out)
  }, integer(1))

  # it's possible for multiple statements to live on the
  # same line (e.g. `print("a"); print("b")`) so only keep
  # uniques
  lines <- unique(lines)

  # convert from lines to ranges (be sure to handle the zero-length case)
  ranges <- list()
  if (length(lines)) {

    if(py_version() >= "3.8") {
      # end_lineno attr only introduced in 3.8
      ends <- lines
      starts <- c(1L, ends[-length(ends)] + 1L)
    } else {
      starts <- lines
      ends <- c(lines[-1] - 1, length(code))
    }
    ranges <- mapply(c, starts, ends, SIMPLIFY = FALSE)
  }

  # Stash some options.
  is_hold <- identical(options$results, "hold")
  is_include <- isTRUE(options$include)
  jupyter_compat <- isTRUE(options$jupyter_compat)

  # line index from which source should be emitted
  pending_source_index <- 1

  # whether an error occurred during execution
  had_error <- FALSE

  # actual outputs to be returned to knitr
  outputs <- stack()

  # 'held' outputs, to be appended at the end (for results = "hold")
  held_outputs <- stack()

  # Outputs to be appended to; these depend on the "hold" option.
  outputs_target <- if (is_hold) held_outputs else outputs

  # synchronize state R -> Python
  eng_python_synchronize_before(options)

  # determine if we should capture errors
  # (don't capture errors during knit)
  capture_errors <-
    identical(options$error, TRUE) ||
    identical(getOption("knitr.in.progress", default = FALSE), FALSE)

  if(isFALSE(options$warning)) {
    py_catch_warnings_ctxt <-
      # need to set record = TRUE, otherwise custom implementations of
      # `warning.showwarning()` leak warnings out of the context.
      import("warnings", convert = FALSE)$catch_warnings(record = TRUE)
    py_catch_warnings_ctxt$`__enter__`()
    on.exit({
      py_catch_warnings_ctxt$`__exit__`(NULL, NULL, NULL)
    }, add = TRUE)
  }

  for (i in seq_along(ranges)) {

    # extract range
    range <- ranges[[i]]
    last_range <- i == length(ranges)

    # extract code to be run
    snippet <- extract(code, range)

    # clear the last value object (so we can tell if it was updated)
    py_compile_eval("'__reticulate_placeholder__'")

    # use trailing semicolon to suppress output of return value
    suppress <- grepl(";\\s*$", snippet) || (jupyter_compat & !last_range)
    compile_mode <- if (suppress) "exec" else "single"

    # run code and capture output
    captured_stdout <- if (capture_errors) {
      tryCatch(
        py_compile_eval(snippet, compile_mode),
        error = function(e) {

          # if the chunk option is error = FALSE (the default).
          # we'll need to bail and not evaluate to the next python expression.
          if (identical(options$error, FALSE))
            had_error <- TRUE

          # format the exception object
          etype <- py_get_attr(e, "__class__")
          traceback <- import("traceback")
          paste0(traceback$format_exception_only(etype, e),
                 collapse = "")
        }
      )

    }
    else
      py_compile_eval(snippet, compile_mode)

    # handle matplotlib plots and other special output
    captured <- eng_python_autoprint(
      captured = captured_stdout,
      options  = options
    )

    # A trailing ';' suppresses output.
    # In jupyter mode, only the last expression in a chunk has repr() output.
    if (suppress)
      captured <- captured_stdout

    # emit outputs if we have any
    has_outputs <-
      !.engine_context$pending_plots$empty() ||
      !identical(captured, "")

    if (has_outputs) {

      # append pending source to outputs (respecting 'echo' option)
      if (!identical(options$echo, FALSE) && !is_hold) {
        extracted <- extract(code, c(pending_source_index, range[2]))
        if(!identical(options$collapse, TRUE) &&
           identical(options$strip.white, TRUE)) {
          extracted <- sub("^\\n+", "", sub("\\n+$", "", extracted))
          # trimws(whitespace = ) requires R 3.6
          # extracted <- trimws(extracted, whitespace = "[\n]")
        }
        output <- structure(list(src = extracted), class = "source")
        outputs$push(output)
      }

      # append captured outputs (respecting 'include' option)
      if (is_include) {
        # append captured output
        if (!identical(captured, ""))
          outputs_target$push(captured)

        # append captured images / figures
        for (plot in .engine_context$pending_plots$data())
          outputs_target$push(plot)
        .engine_context$pending_plots$clear()
      }

      # update pending source range
      pending_source_index <- range[2] + 1

      # bail if we had an error with 'error=FALSE'
      if (had_error && identical(options$error, FALSE))
        break

    }
  }

  # if we have leftover input, add that now
  has_leftovers <-
    !had_error &&
    !identical(options$echo, FALSE) &&
    !identical(options$results, "hold") &&
    pending_source_index <= n

  if (has_leftovers) {
    leftover <- extract(code, c(pending_source_index, n))
    output <- structure(list(src = leftover), class = "source")
    outputs$push(output)
  }

  # check if we need to call matplotlib.pyplot.show()
  # for any pending undisplayed plots
  if(isTRUE(.globals$matplotlib_initialized)) {
    plt <- import("matplotlib.pyplot")
    if(length(plt$get_fignums()))
      plt$show()
  }

  for (plot in .engine_context$pending_plots$data())
    outputs_target$push(plot)
  .engine_context$pending_plots$clear()


  # if we were using held outputs, we just inject the source in now
  if (is_hold) {
    output <- structure(list(src = code), class = "source")
    outputs$push(output)
  }

  # if we had held outputs, add those in now (merging text output as appropriate)
  text_output <- character()

  held_outputs <- held_outputs$data()
  for (i in seq_along(held_outputs)) {

    output <- held_outputs[[i]]
    if (!is.object(output) && is.character(output)) {

      # merge text output and save for later
      text_output <- c(text_output, held_outputs[[i]])

    } else {

      # add in pending text output
      if (length(text_output)) {
        output <- paste(text_output, collapse = "")
        outputs$push(output)
        text_output <- character()
      }

      # add in this piece of output
      outputs$push(held_outputs[[i]])
    }

  }

  # if we have any leftover held output, add in now
  if (length(text_output)) {
    output <- paste(text_output, collapse = "")
    outputs$push(output)
  }

  eng_python_synchronize_after()

  wrap <- getOption("reticulate.engine.wrap", eng_python_wrap)
  wrap(outputs$data(), options)

}

eng_python_initialize <- function(options, envir) {

  if (is.character(options$engine.path))
    use_python(options$engine.path[[1]])

  ensure_python_initialized()
  eng_python_initialize_hooks(options, envir)

}

eng_python_knit_include_graphics <-
  function(options, suffix = NULL, write_figure = function(path) NULL) {

  # ensure that both the figure file saving code, as well as
  # knitr::include_graphics(), are run with the correct working directory.

  # we need to work in either base.dir or output.dir, depending
  # on which of the two has been requested by the user. (note
  # that output.dir should always be set)
  dir <-
    knitr::opts_knit$get("base.dir") %||%
    knitr::opts_knit$get("output.dir")

  # move to the requested directory
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  owd <- setwd(dir)
  on.exit(setwd(owd), add = TRUE)

  # construct plot path
  plot_counter <- yoink("knitr", "plot_counter")
  number <- plot_counter()
  paths <- knitr::fig_path(
    suffix  = suffix %||% options$dev,
    options = options,
    number  = number
  )

  for (path in paths) {
    # ensure parent path exists
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

    # write figures
    write_figure(path)
  }

  # include the first requested path
  knitr::include_graphics(paths[1])

}

eng_python_matplotlib_show <- function(plt, options) {

  on.exit(plt$close())

  # save figure file, return knitr::include_graphics() wrapped figure path
  eng_python_knit_include_graphics(
    options, write_figure = function(path) {
      # save the current figure to all requested devices
      plt$savefig(path, dpi = options$dpi)
    }
  )

}

eng_python_initialize_hooks <- function(options, envir) {

  # set up hooks for matplotlib modules
  matplotlib_modules <- c(
    "matplotlib.artist",
    "matplotlib.pyplot",
    "matplotlib.pylab"
  )

  for (module in matplotlib_modules) {
    py_register_load_hook(module, function(...) {
      eng_python_initialize_matplotlib(options, envir)
    })
  }

  # set up hooks for plotly modules
  plotly_modules <- c(
    "plotly.io",
    "plotlyjs"
  )

  for (module in plotly_modules) {
    py_register_load_hook(module, function(...) {
      eng_python_initialize_plotly(options, envir)
    })
  }

}

eng_python_initialize_matplotlib <- function(options, envir) {

  # early exit if we already initialized
  # (this onload hook is registered for multiple matplotlib submodules)
  if (identical(.globals$matplotlib_initialized, TRUE))
    return(TRUE)

  # attempt to enforce a non-Qt matplotlib backend. this is especially important
  # with RStudio Desktop as attempting to use a Qt backend will cause issues due
  # to mismatched Qt versions between RStudio and Anaconda environments, and
  # will cause crashes when attempting to generate plots
  testthat <- Sys.getenv("TESTTHAT", unset = NA)
  if (is_rstudio_desktop() || identical(testthat, "true")) {

    matplotlib <- import("matplotlib", convert = TRUE)

    # check to see if a backend has already been initialized. if so, we
    # need to switch backends; otherwise, we can simply request to use a
    # specific one when the backend is initialized later
    sys <- import("sys", convert = FALSE)
    if ("matplotlib.backends" %in% names(sys$modules)) {
      matplotlib$pyplot$switch_backend("agg")
    } else {
      version <- numeric_version(matplotlib$`__version__`)
      if (version < "3.3.0")
        matplotlib$use("agg", warn = FALSE, force = TRUE)
      else
        matplotlib$use("agg", force = TRUE)
    }
  }

  # double-check that we can load 'pyplot' (this can fail if matplotlib
  # is installed but is initialized to a backend missing some required components)
  if (!py_module_available("matplotlib.pyplot"))
    return()

  plt <- import("matplotlib.pyplot", convert = FALSE)

  # set up figure dimensions
  plt$rc("figure", figsize = tuple(options$fig.width, options$fig.height))

  # override show implementation
  plt$show <- function(...) {

    # get current chunk options
    options <- knitr::opts_current$get()

    # call hook to generate plot
    hook <- getOption("reticulate.engine.matplotlib.show", eng_python_matplotlib_show)
    graphic <- hook(plt, options)

    # update set of pending plots
    .engine_context$pending_plots$push(graphic)

    # return None to ensure no printing of output here (just inclusion of
    # plot as a side effect)
    py_none()

  }

  .globals$matplotlib_initialized <- TRUE

}

eng_python_initialize_plotly <- function(options, envir) {

  # mark initialization done
  if (identical(.globals$plotly_initialized, TRUE))
    return(TRUE)

  .globals$plotly_initialized <- TRUE

  # override the figure 'show' method to just return the plot object itself
  # the auto-printer will then handle rendering the image as appropriate
  io <- import("plotly.io", convert = FALSE)
  io$show <- function(self, ...) self

  renderers <- io$renderers
  if (!py_bool(renderers$default))
    renderers$default <- "plotly_mimetype+notebook"

}

# synchronize objects R -> Python
eng_python_synchronize_before <- function(options) {
  py_inject_r()
  if(isTRUE(.globals$matplotlib_initialized)) {

    # set up figure dimensions
    plt <- import("matplotlib.pyplot")
    plt$rc("figure", figsize = tuple(options$fig.width, options$fig.height))
  }
}

# synchronize objects Python -> R
eng_python_synchronize_after <- function() {}

eng_python_wrap <- function(outputs, options) {
  knitr::engine_output(options, out = outputs)
}

eng_python_validate_options <- function(options) {

  # warn about unsupported numeric options and convert to TRUE
  no_numeric <- c("eval", "echo", "warning")
  for (option in no_numeric) {
    if (is.numeric(options[[option]])) {
      fmt <- "numeric '%s' chunk option not supported by reticulate engine"
      msg <- sprintf(fmt, option)
      warning(msg, call. = FALSE)
      options[[option]] <- TRUE
    }
  }

  options
}

eng_python_is_matplotlib_output <- function(value) {

  matplotlib_plot_types <- c("matplotlib.artist.Artist",
                             "matplotlib.container.Container",
                             "matplotlib.image.AxesImage",
                             "matplotlib.image.BboxImage",
                             "matplotlib.image.FigureImage",
                             "matplotlib.image.NonUniformImage",
                             "matplotlib.image.PcolorImage")

  if (inherits(value, c("python.builtin.tuple", "python.builtin.list")) &&
      length(value) > 0L) {

    # some functions returned list-"boxed" images, like [<img>]
    if (inherits(py_get_item(value, 0L), matplotlib_plot_types))
      return(TRUE)

    # plt.hist returns (<np.array>, <np.array>, <img>)
    if(length(value) > 1L &&
       inherits(py_get_item(value, length(value)-1L), matplotlib_plot_types))
      return(TRUE)
  }

  inherits(value, matplotlib_plot_types)
}

eng_python_is_seaborn_output <- function(value) {
  inherits(value, "seaborn.axisgrid.Grid")
}

eng_python_is_plotly_plot <- function(value) {
  inherits(value, "plotly.basedatatypes.BaseFigure")
}

eng_python_is_altair_chart <- function(value) {

  # support different API versions, assuming that the class name
  # otherwise remains compatible
  classes <- class(value)
  pattern <- "^altair\\.vegalite\\.v[[:digit:]]+\\.api\\.(HConcat|VConcat|Layer|Repeat|Facet)?Chart$"
  any(grepl(pattern, classes))

}

eng_python_altair_chart_id <- function(options, ids) {

  label <- options$label
  components <- c(label, "altair-viz")
  if (exists(label, envir = ids)) {
    id <- get(label, envir = ids)
    components <- c(components, id + 1)
    assign(label, id + 1, envir = ids)
  } else {
    assign(label, 1L, envir = ids)
  }

  paste(components, collapse = "-")

}

eng_python_autoprint <- function(captured, options) {

  # bail if no new value was produced by interpreter
  value <- py_last_value()
  if (py_is_none(value))
    return(captured)

  # ignore placeholder outputs
  if (inherits(value, "python.builtin.str")) {
    contents <- py_to_r(value)
    if (identical(contents, "__reticulate_placeholder__"))
      return(captured)
  }

  # check if output format is html
  isHtml <- knitr::is_html_output()

  if (eng_python_is_matplotlib_output(value)) {
    # We handle pending Matplotlib plots with fignums check later.

    # Always suppress Matplotlib reprs
    return("")

  } else if (eng_python_is_seaborn_output(value)) {

    # get figure path
    included_path <- eng_python_knit_include_graphics(
      options, write_figure = function(path) {
      value$savefig(path)
  })

    .engine_context$pending_plots$push(included_path)
    return("")

  } else if (inherits(value, "pandas.core.frame.DataFrame")) {

    return(captured)

  } else if (isHtml && py_has_method(value, "_repr_html_")) {

    py_capture_output({
      data <- as_r_value(value$`_repr_html_`())
    })
    .engine_context$pending_plots$push(knitr::raw_html(data))
    return("")

  } else if (eng_python_is_plotly_plot(value) &&
             py_module_available("psutil") &&
             py_module_available("kaleido")) {

    included_path <- eng_python_knit_include_graphics(
      options, write_figure = function(path) {
        value$write_image(
          file   = path,
          width  = options$out.width.px,
          height = options$out.height.px
        )
      }
    )
    .engine_context$pending_plots$push(included_path)
    return("")

  } else if (eng_python_is_altair_chart(value)) {

    # set width if it's not already set
    # This only applies to Chart objects, compound charts like HConcatChart
    # don't have a 'width' or 'height' property attribute.
    # TODO: add support for propagating width/height options from knitr to
    # altair compound charts
    width <- py_get_attr(value, "width", TRUE)
    if (inherits(width, "altair.utils.schemapi.UndefinedType")) {
      width <- options$altair.fig.width %||% options$out.width.px %||% 810L
      value <- value$properties(width = width)
    }

    # set height if it's not already set
    height <- py_get_attr(value, "height", TRUE)
    if (inherits(height, "altair.utils.schemapi.UndefinedType")) {
      height <- options$altair.fig.height %||% options$out.height.px %||% 400L
      value <- value$properties(height = height)
    }

    # set a unique id (used for div container for figure)
    id <- eng_python_altair_chart_id(options, .engine_context$altair_ids)

    # convert to HTML or PNG as appropriate
    if (isHtml) {
      data <- as_r_value(value$to_html(output_div = id))
      .engine_context$pending_plots$push(knitr::raw_html(data))
    } else {

      included_path <- eng_python_knit_include_graphics(
        options, write_figure = function(path) {
          value$save(path)
        }
      )
      .engine_context$pending_plots$push(included_path)
    }

    return("")

  } else if (py_has_method(value, "_repr_markdown_")) {

    data <- as_r_value(value$`_repr_markdown_`())
    .engine_context$pending_plots$push(knitr::asis_output(data))
    return("")

  } else if (py_has_method(value, "to_html")) {

    data <- as_r_value(value$to_html())
    .engine_context$pending_plots$push(knitr::raw_html(data))
    return("")

  } else {

    # nothing special to do
    return(captured)

  }

}
