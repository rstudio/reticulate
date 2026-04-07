#ifndef RETICULATE_R_API_H
#define RETICULATE_R_API_H

inline SEXP reticulate_find_namespace(const char* name) {
  SEXP ns = R_NilValue;
  SEXP name_sexp = PROTECT(Rf_mkString(name));
  ns = R_FindNamespace(name_sexp);
  UNPROTECT(1);
  return ns;
}

// R_getVarEx() is the supported replacement for Rf_findVar* on R >= 4.5.
// It is not identical: it forces promises and errors on R_MissingArg.
// Use these wrappers only where those differences are acceptable.
inline SEXP reticulate_get_var_or_null(SEXP env, SEXP sym) {
#if defined(R_VERSION) && R_VERSION >= R_Version(4, 5, 0)
  return R_getVarEx(sym, env, FALSE, NULL);
#else
  SEXP value = Rf_findVarInFrame(env, sym);
  return value == R_UnboundValue ? NULL : value;
#endif
}

inline SEXP reticulate_get_var(SEXP sym, SEXP env) {
#if defined(R_VERSION) && R_VERSION >= R_Version(4, 5, 0)
  SEXP value = R_getVarEx(sym, env, FALSE, NULL);
#else
  SEXP value = Rf_findVarInFrame(env, sym);
  if (value == R_UnboundValue)
    value = NULL;
#endif
  if (value == NULL)
    Rf_error("object '%s' not found", CHAR(PRINTNAME(sym)));
  return value;
}

#endif
