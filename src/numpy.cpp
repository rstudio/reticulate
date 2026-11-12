#define RCPP_NO_MODULES
#define RCPP_NO_SUGAR
#include <Rcpp.h>
#include <Rversion.h>

#if R_VERSION >= R_Version(4, 3, 0)
#include <R_ext/Altrep.h>

static R_altrep_class_t numpy_column_sources_class;

static R_xlen_t numpy_sources_length(SEXP x) {
  return static_cast<R_xlen_t>(REAL(VECTOR_ELT(R_altrep_data1(x), 1))[0]);
}

static SEXP numpy_sources_value(SEXP x) {
  return VECTOR_ELT(R_altrep_data1(x), 0);
}

static SEXP numpy_sources_new(SEXP source, R_xlen_t size) {
  // An empty column or prototype must not keep the source array alive.
  if (size == 0)
    return Rf_allocVector(VECSXP, 0);

  SEXP state = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(state, 0, source);
  SET_VECTOR_ELT(state, 1, Rf_ScalarReal(static_cast<double>(size)));
  SEXP out = R_new_altrep(numpy_column_sources_class, state, R_NilValue);
  UNPROTECT(1);
  return out;
}

static SEXP numpy_sources_elt(SEXP x, R_xlen_t i) {
  SEXP data = R_altrep_data2(x);
  if (data != R_NilValue)
    return VECTOR_ELT(data, i);
  return numpy_sources_value(x);
}

static SEXP numpy_sources_materialize(SEXP x) {
  SEXP data = R_altrep_data2(x);
  if (data != R_NilValue)
    return data;

  R_xlen_t size = numpy_sources_length(x);
  data = PROTECT(Rf_allocVector(VECSXP, size));
  SEXP source = numpy_sources_value(x);
  for (R_xlen_t i = 0; i < size; ++i)
    SET_VECTOR_ELT(data, i, source);
  R_set_altrep_data2(x, data);
  // The materialized slots now own the references. Drop the extra reference
  // without changing the compact state shared by any duplicates.
  SEXP state = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(state, 1, VECTOR_ELT(R_altrep_data1(x), 1));
  R_set_altrep_data1(x, state);
  UNPROTECT(2);
  return data;
}

static void numpy_sources_set_elt(SEXP x, R_xlen_t i, SEXP value) {
  SET_VECTOR_ELT(numpy_sources_materialize(x), i, value);
}

static void* numpy_sources_dataptr(SEXP x, Rboolean writeable) {
#if R_VERSION >= R_Version(4, 6, 0)
  return DATAPTR_RW(numpy_sources_materialize(x));
#else
  return DATAPTR(numpy_sources_materialize(x));
#endif
}

static const void* numpy_sources_dataptr_or_null(SEXP x) {
  SEXP data = R_altrep_data2(x);
  return data == R_NilValue ? NULL : DATAPTR_OR_NULL(data);
}

static SEXP numpy_sources_duplicate(SEXP x, Rboolean deep) {
  SEXP data = R_altrep_data2(x);
  // The ndarray is a shared Python reference, but list edits obey R's
  // copy-on-modify semantics after the source list has been materialized.
  data = PROTECT(deep ? Rf_duplicate(data) : Rf_shallow_duplicate(data));
  SEXP out = R_new_altrep(numpy_column_sources_class, R_altrep_data1(x), data);
  UNPROTECT(1);
  return out;
}

static bool numpy_source_location_valid(SEXP index, R_xlen_t i, R_xlen_t size) {
  double location = TYPEOF(index) == INTSXP ? INTEGER_ELT(index, i) : REAL_ELT(index, i);
  return location >= 1 && location <= size;
}

static SEXP numpy_sources_subset(SEXP x, SEXP index, SEXP call) {
  if (R_altrep_data2(x) != R_NilValue)
    return NULL;

  R_xlen_t size = numpy_sources_length(x);
  R_xlen_t n = XLENGTH(index);
  R_xlen_t i = 0;
  for (; i < n; ++i) {
    if (!numpy_source_location_valid(index, i, size))
      break;
  }
  if (i == n)
    return numpy_sources_new(numpy_sources_value(x), n);

  // Base R and vctrs use NULL list elements for missing and out-of-range rows.
  SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
  SEXP source = numpy_sources_value(x);
  for (R_xlen_t j = 0; j < n; ++j) {
    if (numpy_source_location_valid(index, j, size))
      SET_VECTOR_ELT(out, j, source);
  }
  UNPROTECT(1);
  return out;
}

#endif

// [[Rcpp::init]]
void numpy_column_init(DllInfo* dll) {
#if R_VERSION >= R_Version(4, 3, 0)
  numpy_column_sources_class = R_make_altlist_class("numpy_column_sources", "reticulate", dll);
  R_set_altrep_Length_method(numpy_column_sources_class, numpy_sources_length);
  R_set_altrep_Duplicate_method(numpy_column_sources_class, numpy_sources_duplicate);
  R_set_altlist_Elt_method(numpy_column_sources_class, numpy_sources_elt);
  R_set_altlist_Set_elt_method(numpy_column_sources_class, numpy_sources_set_elt);
  R_set_altvec_Dataptr_method(numpy_column_sources_class, numpy_sources_dataptr);
  R_set_altvec_Dataptr_or_null_method(numpy_column_sources_class, numpy_sources_dataptr_or_null);
  R_set_altvec_Extract_subset_method(numpy_column_sources_class, numpy_sources_subset);
#endif
}

// [[Rcpp::export]]
SEXP numpy_column_sources(SEXP source, int size) {
#if R_VERSION >= R_Version(4, 3, 0)
  return numpy_sources_new(source, size);
#else
  Rcpp::stop("NumPy dataframe columns require R 4.3.0 or newer");
#endif
}

// [[Rcpp::export]]
SEXP numpy_column_source(SEXP sources) {
#if R_VERSION >= R_Version(4, 3, 0)
  if (ALTREP(sources) && R_altrep_inherits(sources, numpy_column_sources_class) &&
      R_altrep_data2(sources) == R_NilValue)
    return numpy_sources_value(sources);
#endif
  return R_NilValue;
}
