
#' Set Python and NumPy random seeds
#'
#' Set various random seeds required to ensure reproducible results. The
#' provided `seed` value will establish a new random seed for Python and NumPy,
#' and will also (by default) disable hash randomization.
#'
#' @param seed A single value, interpreted as an integer
#' @param disable_hash_randomization Disable hash randomization, which is
#'   another common source of variable results. See
#'   <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>
#'
#' @details This function does not set the R random seed, for that you
#'   should call [set.seed()].
#'
#' @export
py_set_seed <- function(seed, disable_hash_randomization = TRUE) {

  # cast to integer
  seed <- as.integer(seed)

  # Ensure reproducibility for certain hash-based operations for Python 3
  # References: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
  #             https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
  if (disable_hash_randomization) {
    os <- import("os")
    Sys.setenv(PYTHONHASHSEED = "0")
    os$environ[["PYTHONHASHSEED"]] <- "0"
  }

  # set Python python random seed
  random <- import("random")
  random$seed(seed)

  # set numpy seed if numpy is available
  if (py_numpy_available()) {
    np <- import("numpy")
    np$random$seed(seed)
  }

  invisible(NULL)
}



