
#ifndef RETICULATE_H
#define RETICULATE_H

#ifndef DBG
# define DBG Rprintf
#endif

#ifndef LibExtern
# ifdef _WIN32
#  define LibExtern __declspec(dllimport) extern
# else
#  define LibExtern extern
# endif
#endif

extern "C" {
extern void Rprintf(const char*, ...);  
}

#endif /* RETICULATE_H */
