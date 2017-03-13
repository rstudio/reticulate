

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern "C" void R_init_reticulate(DllInfo* info) {
  R_registerRoutines(info, NULL, NULL, NULL, NULL);
  R_useDynamicSymbols(info, TRUE);
}