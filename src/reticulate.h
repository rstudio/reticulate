
#ifndef RETICULATE_H
#define RETICULATE_H

#include <R_ext/Boolean.h>

// Debug macros.

#define DBG(...)
// #define DBG(fmt, ...) Rprintf("[DEBUG] (%s:%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

// For importing symbols from R.
// Forget to include __declspec(dllimport) at your own peril.
#ifndef LibExtern
# ifdef _WIN32
#  define LibExtern __declspec(dllimport) extern
# else
#  define LibExtern extern
# endif
#endif

// Forward declarations for some R functions that we'd like to avoid
// pulling in all R headers for.
extern "C" {
extern Rboolean R_ToplevelExec(void (*func)(void*), void*);
extern void R_ProcessEvents();
extern void Rprintf(const char*, ...);  
}

#endif /* RETICULATE_H */
