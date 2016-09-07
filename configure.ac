# Process this file with autoconf to produce a configure script.
# This file is based on the configure.ac distributed with the rPython package

AC_INIT(tensorflow, 0.1, jj@rstudio.com)
AC_CONFIG_SRCDIR([src])

# find R home and set CC/CFLAGS
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "Could not determine R_HOME"
  exit 1
fi
RBIN="${R_HOME}/bin/R"
CC=`"${RBIN}" CMD config CC`;
CFLAGS=`"${RBIN}" CMD config CFLAGS`

RLD=`"${RBIN}" CMD config --ldflags 2>/dev/null`
has_R_shlib=no
if test -n "$RLD"; then
   has_R_shlib=yes
fi

PYTHONCONFIG=python-config
PYTHON=python

# Extract the linker and include flags for python
: ${PYTHONCONFIG=`which ${PYTHONCONFIG}`}
if test -z "${PYTHONCONFIG}"; then
  echo "Could not locate ${PYTHONCONFIG}"
  exit 1
fi

PKG_LIBS=`$PYTHONCONFIG --ldflags`
PKG_CPPFLAGS=`$PYTHONCONFIG --includes`

# find python
: ${PYTHON=`which ${PYTHON}`}
if test -z "${PYTHON}"; then
  echo "Could not locate ${PYTHON}"
  exit 1
fi

# find the numpy includes
NUMPY_INCLUDE_DIR=`$PYTHON -c "import numpy; print(numpy.get_include())"`
if test -z "${NUMPY_INCLUDE_DIR}"; then
  echo "Could not locate numpy include directory"
  exit 1
fi
PKG_CPPFLAGS="$PKG_CPPFLAGS -I$NUMPY_INCLUDE_DIR"

# Find the .so library for python
PYTHONLIBFILE=libpython`$PYTHON -c "import sys; print(sys.version[[:3]])"`.so
PKG_CPPFLAGS="$PKG_CPPFLAGS -D PYTHONLIBFILE=$PYTHONLIBFILE"

AC_SUBST(PKG_LIBS)
AC_SUBST(PKG_CPPFLAGS)

AC_PREREQ

# Checks for programs.
AC_PROG_CC

# Checks for libraries.
# (None)

# Checks for header files.
# (None)

# Checks for typedefs, structures, and compiler characteristics.
# (None)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
