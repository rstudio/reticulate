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
  echo "Using python binary from PATH"
  USE_PYTHON=python
else
  TENSORFLOW_PYTHON="${TENSORFLOW_PYTHON/#\~/$HOME}"
  echo "Using python binary at ${TENSORFLOW_PYTHON}"
  USE_PYTHON=${TENSORFLOW_PYTHON}
fi

# find tensorflow target python version
if test -z "${TENSORFLOW_PYTHON_VERSION}"; then
  # use 3 if the default python binary aliases python3
  PYTHON_VERSION=`${USE_PYTHON} -c 'import sys; import platform; major, minor, patch = platform.python_version_tuple(); sys.stdout.write(major);'`
  if [[ $PYTHON_VERSION == "3" ]]; then
    USE_PYTHON="${USE_PYTHON}3"
  fi
else
  echo "Using python version ${TENSORFLOW_PYTHON_VERSION}"
  USE_PYTHON=${USE_PYTHON}${TENSORFLOW_PYTHON_VERSION}
fi

# find python
PYTHON_BIN=`which ${USE_PYTHON}`
if test -z "${PYTHON_BIN}"; then
  echo "Could not locate ${PYTHON_BIN}"
  exit 1
fi

# Check for Anaconda and error out in this case
# (see https://github.com/ContinuumIO/anaconda-issues/issues/498)
PYTHON_VERSION=`${PYTHON_BIN} -c 'import sys; sys.stdout.write(sys.version);'`
case "${PYTHON_VERSION}" in
  *Anaconda*)  CONDA_PYTHON=1 ;;
  *Continuum*) CONDA_PYTHON=1 ;;
  *)           CONDA_PYTHON=0 ;;
esac
if test "${CONDA_PYTHON}" -eq "1"
then
  echo ""
  echo " The tensorflow R package is not compatible with "
  echo " Conda/Anaconda distributions of Python. Please "
  echo " use another version of Python (you can select a "
  echo " version to compile against by defining the "
  echo " TENSORFLOW_PYTHON environment variable)."
  echo ""
  exit 1
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
PYTHON_PREFIX=`$PYTHONCONFIG --prefix`

# find the numpy includes
NUMPY_INCLUDE_DIR=`$PYTHON_BIN -c "import numpy; print(numpy.get_include())"`
if test -z "${NUMPY_INCLUDE_DIR}"; then
  echo "Could not locate numpy include directory"
  exit 1
fi
PKG_CPPFLAGS="$PKG_CPPFLAGS -I$NUMPY_INCLUDE_DIR"

# Find the .so library for python
PYTHONLIBFILE=libpython`$PYTHON_BIN -c "import sys; print(sys.version[[:3]])"`.so
PKG_CPPFLAGS="$PKG_CPPFLAGS -D PYTHONLIBFILE=$PYTHONLIBFILE"

AC_SUBST(PYTHON_BIN)
AC_SUBST(PKG_LIBS)
AC_SUBST(PKG_CPPFLAGS)
AC_SUBST(NUMPY_INCLUDE_DIR)
AC_SUBST(PYTHON_PREFIX)

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
