#ifndef RETICULATE_R_API_H
#define RETICULATE_R_API_H

// R_getVarEx() is the supported replacement for Rf_findVar* on R >= 4.5.
// It is not identical: it forces promises and errors on R_MissingArg.
// Use these wrappers only where those differences are acceptable.
inline SEXP reticulate_get_var_in_frame(SEXP env, SEXP sym) {
#if defined(R_VERSION) && R_VERSION >= R_Version(4, 5, 0)
  return R_getVarEx(sym, env, FALSE, R_UnboundValue);
#else
  return Rf_findVarInFrame(env, sym);
#endif
}

inline SEXP reticulate_get_var(SEXP sym, SEXP env) {
#if defined(R_VERSION) && R_VERSION >= R_Version(4, 5, 0)
  return R_getVarEx(sym, env, TRUE, R_UnboundValue);
#else
  return Rf_findVar(sym, env);
#endif
}

#endif
