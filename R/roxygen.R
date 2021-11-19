
#' @param packages A character vector, indicating package names which should be
#'   installed or removed. Use `python=<version>` to request the installation
#'   of a specific version of Python.
#' 
#' @param envname The name of, or path to, a conda environment.
#' 
#' @param conda The path to a `conda` executable. Use `"auto"` to allow
#'   `reticulate` to automatically find an appropriate `conda` binary. See
#'   [conda_binary()] for more details on how `reticulate` tries to resolve
#'   the `conda` executable.
#'
#' @param forge Boolean; include the [conda-forge](https://conda-forge.org/)
#'   repository?
#'   
#' @param channel An optional character vector of conda channels to include.
#'   When specified, the `forge` argument is ignored. If you need to
#'   specify multiple channels, including the conda Forge, you can use
#'   `c("conda-forge", <other channels>)`.
#'   
#' @param ... Optional arguments, reserved for future expansion.
#'
#' @name conda-params
NULL
