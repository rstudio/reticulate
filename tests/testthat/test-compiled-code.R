context("compiled code")

test_that("compiled code avoids deprecated variable lookup entry points", {
  skip_if(getRversion() < "4.5.0")
  skip_if(.Platform$OS.type == "windows")

  nm <- Sys.which("nm")
  skip_if(nm == "")

  dylib <- getLoadedDLLs()[["reticulate"]][["path"]]
  symbols <- system2(nm, c("-u", dylib), stdout = TRUE, stderr = TRUE)

  expect_false(any(grepl("\\b_?Rf_findVar(InFrame)?$", symbols)))
})
