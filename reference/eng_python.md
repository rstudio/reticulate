# A reticulate Engine for Knitr

This provides a `reticulate` engine for `knitr`, suitable for usage when
attempting to render Python chunks. Using this engine allows for shared
state between Python chunks in a document – that is, variables defined
by one Python chunk can be used by later Python chunks.

## Usage

``` r
eng_python(options)
```

## Arguments

- options:

  Chunk options, as provided by `knitr` during chunk execution.

## Details

The engine can be activated by setting (for example)

    knitr::knit_engines$set(python = reticulate::eng_python)

Typically, this will be set within a document's setup chunk, or by the
environment requesting that Python chunks be processed by this engine.
Note that `knitr` (since version 1.18) will use the `reticulate` engine
by default when executing Python chunks within an R Markdown document.

## Supported `knitr` chunk options

For most options, reticulate's python engine behaves the same as the
default R engine included in knitr, but they might not support all the
same features. Options in *italic* are equivalent to knitr, but with
modified behavior.

- *`eval`* (`TRUE`, logical): If `TRUE`, all expressions in the chunk
  are evaluated. If `FALSE`, no expression is evaluated. Unlike knitr's
  R engine, it doesn't support numeric values indicating the expressions
  to evaluate.

- *`echo`* (`TRUE`, logical): Whether to display the source code in the
  output document. Unlike knitr's R engine, it doesn't support numeric
  values indicating the expressions to display.

- `results` (`'markup'`, character): Controls how to display the text
  results. Note that this option only applies to normal text output (not
  warnings, messages, or errors). The behavior should be identical to
  knitr's R engine.

- `collapse` (`FALSE`, logical): Whether to, if possible, collapse all
  the source and output blocks from one code chunk into a single block
  (by default, they are written to separate blocks). This option only
  applies to Markdown documents.

- `error` (`TRUE`, logical): Whether to preserve errors. If `FALSE`
  evaluation stops on errors. (Note that RMarkdown sets it to `FALSE`).

- *`warning`* (`TRUE`, logical): Whether to preserve warnings in the
  output. If FALSE, all warnings will be suppressed. Doesn't support
  indices.

- `include` (`TRUE`, logical): Whether to include the chunk output in
  the output document. If `FALSE`, nothing will be written into the
  output document, but the code is still evaluated and plot files are
  generated if there are any plots in the chunk, so you can manually
  insert figures later.

- `dev`: The graphical device to generate plot files. See knitr
  documentation for additional information.

- `base.dir` (`NULL`; character): An absolute directory under which the
  plots are generated.

- `strip.white` (TRUE; logical): Whether to remove blank lines in the
  beginning or end of a source code block in the output.

- `dpi` (72; numeric): The DPI (dots per inch) for bitmap devices (dpi
  \* inches = pixels).

- `fig.width`, `fig.height` (both are 7; numeric): Width and height of
  the plot (in inches), to be used in the graphics device.

- `label`: The chunk label for each chunk is assumed to be unique within
  the document. This is especially important for cache and plot
  filenames, because these filenames are based on chunk labels. Chunks
  without labels will be assigned labels like unnamed-chunk-i, where i
  is an incremental number.

### Python engine only options

- **`jupyter_compat`** (FALSE, logical): If `TRUE` then, like in Jupyter
  notebooks, only the last expression in the chunk is printed to the
  output.

- **`out.width.px`**, **`out.height.px`** (810, 400, both integers):
  Width and height of the plot in the output document, which can be
  different with its physical `fig.width` and `fig.height`, i.e., plots
  can be scaled in the output document. Unlike knitr's `out.width`, this
  is always set in pixels.

- **`altair.fig.width`**, **`altair.fig.height`**: If set, is used
  instead of `out.width.px` and `out.height.px` when writing Altair
  charts.
