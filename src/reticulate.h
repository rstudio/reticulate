
#ifndef RETICULATE_H
#define RETICULATE_H

#define DBG(fmt, ...)
// #define DBG(fmt, ...) Rprintf("[DEBUG] (%s:%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

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
