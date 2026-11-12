#define RCPP_NO_MODULES
#define RCPP_NO_SUGAR
#include <Rcpp.h>
#include <Rversion.h>

#if R_VERSION >= R_Version(4, 3, 0)
#include <R_ext/Altrep.h>

extern SEXP ns_reticulate;

static R_altrep_class_t numpy_array_proxy_class;

static R_xlen_t numpy_proxy_length(SEXP x) {
  return INTEGER(VECTOR_ELT(R_altrep_data1(x), 1))[0];
}

static SEXP numpy_proxy_array(SEXP x) {
  return VECTOR_ELT(R_altrep_data1(x), 0);
}

static SEXP numpy_proxy_elt(SEXP x, R_xlen_t i) {
  SEXP data = R_altrep_data2(x);
  if (data != R_NilValue)
    return VECTOR_ELT(data, i);

  SEXP index = PROTECT(Rf_ScalarInteger(i + 1));
  SEXP call = PROTECT(Rf_lang3(Rf_install("numpy_array_proxy_row"),
                              numpy_proxy_array(x), index));
  SEXP out = Rf_eval(call, ns_reticulate);
  UNPROTECT(2);
  return out;
}

static SEXP numpy_proxy_materialize(SEXP x) {
  SEXP data = R_altrep_data2(x);
  if (data != R_NilValue)
    return data;

  R_xlen_t size = numpy_proxy_length(x);
  data = PROTECT(Rf_allocVector(VECSXP, size));
  for (R_xlen_t i = 0; i < size; ++i) {
    SEXP row = PROTECT(numpy_proxy_elt(x, i));
    SET_VECTOR_ELT(data, i, row);
    UNPROTECT(1);
  }
  // Publish only the complete list: an interrupted evaluation can be retried.
  R_set_altrep_data2(x, data);
  UNPROTECT(1);
  return data;
}

static void numpy_proxy_set_elt(SEXP x, R_xlen_t i, SEXP value) {
  SET_VECTOR_ELT(numpy_proxy_materialize(x), i, value);
}

static void* numpy_proxy_dataptr(SEXP x, Rboolean writeable) {
#if R_VERSION >= R_Version(4, 6, 0)
  return DATAPTR_RW(numpy_proxy_materialize(x));
#else
  return DATAPTR(numpy_proxy_materialize(x));
#endif
}

static const void* numpy_proxy_dataptr_or_null(SEXP x) {
  SEXP data = R_altrep_data2(x);
  return data == R_NilValue ? NULL : DATAPTR_OR_NULL(data);
}

static SEXP numpy_proxy_duplicate(SEXP x, Rboolean deep) {
  SEXP data = R_altrep_data2(x);
  // The ndarray is a shared Python reference, but list edits obey R's
  // copy-on-modify semantics after the proxy has been materialized.
  data = PROTECT(deep ? Rf_duplicate(data) : Rf_shallow_duplicate(data));
  SEXP out = R_new_altrep(numpy_array_proxy_class, R_altrep_data1(x), data);
  UNPROTECT(1);
  return out;
}

static SEXP numpy_proxy_subset(SEXP x, SEXP index, SEXP call) {
  if (R_altrep_data2(x) != R_NilValue)
    return NULL;

  SEXP expr = PROTECT(Rf_lang3(Rf_install("numpy_array_proxy_slice"),
                              numpy_proxy_array(x), index));
  SEXP out = Rf_eval(expr, ns_reticulate);
  UNPROTECT(1);
  return out;
}

#endif

// [[Rcpp::init]]
void numpy_array_proxy_init(DllInfo* dll) {
#if R_VERSION >= R_Version(4, 3, 0)
  numpy_array_proxy_class = R_make_altlist_class("numpy_array_rows", "reticulate", dll);
  R_set_altrep_Length_method(numpy_array_proxy_class, numpy_proxy_length);
  R_set_altrep_Duplicate_method(numpy_array_proxy_class, numpy_proxy_duplicate);
  R_set_altlist_Elt_method(numpy_array_proxy_class, numpy_proxy_elt);
  R_set_altlist_Set_elt_method(numpy_array_proxy_class, numpy_proxy_set_elt);
  R_set_altvec_Dataptr_method(numpy_array_proxy_class, numpy_proxy_dataptr);
  R_set_altvec_Dataptr_or_null_method(numpy_array_proxy_class, numpy_proxy_dataptr_or_null);
  R_set_altvec_Extract_subset_method(numpy_array_proxy_class, numpy_proxy_subset);
#endif
}

// [[Rcpp::export]]
SEXP numpy_array_proxy(SEXP array, int size) {
#if R_VERSION >= R_Version(4, 3, 0)
  SEXP state = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(state, 0, array);
  SET_VECTOR_ELT(state, 1, Rf_ScalarInteger(size));
  SEXP out = R_new_altrep(numpy_array_proxy_class, state, R_NilValue);
  UNPROTECT(1);
  return out;
#else
  Rcpp::stop("NumPy dataframe columns require R 4.3.0 or newer");
#endif
}

// [[Rcpp::export]]
SEXP numpy_array_proxy_data(SEXP proxy) {
#if R_VERSION >= R_Version(4, 3, 0)
  if (ALTREP(proxy) && R_altrep_inherits(proxy, numpy_array_proxy_class) &&
      R_altrep_data2(proxy) == R_NilValue)
    return numpy_proxy_array(proxy);
#endif
  return R_NilValue;
}
