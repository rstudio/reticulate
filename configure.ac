AC_INIT(tensorflow, 0.2, jj@rstudio.com)
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


# find tensorflow target python binary
if test -z "${TENSORFLOW_PYTHON}"
then
  USE_PYTHON=python
else
  TENSORFLOW_PYTHON="${TENSORFLOW_PYTHON/#\~/$HOME}"
  echo "Using python binary at ${TENSORFLOW_PYTHON}"
  USE_PYTHON=${TENSORFLOW_PYTHON}
fi

# find tensorflow target python version
if ! test -z "${TENSORFLOW_PYTHON_VERSION}"
then
  echo "Using python version ${TENSORFLOW_PYTHON_VERSION}"
  USE_PYTHON=${USE_PYTHON}${TENSORFLOW_PYTHON_VERSION}
fi

# determine path to python-config
USE_PYTHON_CONFIG=${USE_PYTHON}-config

# Extract the linker and include flags for python
: ${PYTHONCONFIG=`which ${USE_PYTHON_CONFIG}`}
if test -z "${PYTHONCONFIG}"; then
  echo "Could not locate ${USE_PYTHON_CONFIG}"
  exit 1
fi

PKG_LIBS=`$PYTHONCONFIG --ldflags`
PKG_CPPFLAGS=`$PYTHONCONFIG --includes`

# find python
: ${PYTHON=`which ${USE_PYTHON}`}
PYTHON_BIN=`which ${USE_PYTHON}`
if test -z "${PYTHON}"; then
  echo "Could not locate ${PYTHON}"
  exit 1
fi

# anaconda doesn't provide a -L in ldflags so let's look for one if we need to
# (see https://github.com/JuliaPy/PyCall.jl/issues/37)
if [[ "$PKG_LIBS" != *"-L"* ]]; then
  PYTHON_BIN_DIR=`dirname ${PYTHON_BIN}`
  PYTHON_ROOT_DIR=`dirname ${PYTHON_BIN_DIR}`
  PYTHON_LIB_DIR=${PYTHON_ROOT_DIR}/lib
  PKG_LIBS="${PKG_LIBS} -L${PYTHON_LIB_DIR}"
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

AC_SUBST(PYTHON_BIN)
AC_SUBST(PKG_LIBS)
AC_SUBST(PKG_CPPFLAGS)
AC_SUBST(NUMPY_INCLUDE_DIR)

AC_PREREQ

# Checks for programs.
AC_PROG_CC

# Checks for libraries.
# (None)

# Checks for header files.
# (None)

# Checks for typedefs, structures, and compiler characteristics.
# (None)

AC_CONFIG_FILES([src/Makevars R/config.R])
AC_OUTPUT
