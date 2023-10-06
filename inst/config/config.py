
import platform
import sys
import os

# The 'sysconfig' module is only available with Python 2.7 and newer, but
# an equivalent module in 'distutils' is available for Python 2.6.
if sys.version_info < (2, 7):
  from distutils import sysconfig
else:
  import sysconfig

# The 'imp' module is deprecated since Python 3.4, and the use of
# 'importlib' is recommended instead.
if sys.version_info < (3, 4):
  import imp
  def module_path(name):
    if name in sys.builtin_module_names:
      return "[builtin module]"
    spec = imp.find_module(name)
    return spec[1]
else:
  from importlib import util
  def module_path(name):
    if name in sys.builtin_module_names:
      return "[builtin module]"
    spec = util.find_spec(name)
    origin = spec.origin
    return origin[:origin.rfind('/')]

# Get appropriate path-entry separator for platform
pathsep = ";" if os.name == "nt" else ":"

# Read default configuration values
config = {
  "Architecture"     : platform.architecture()[0],
  "Version"          : str(sys.version).replace("\n", " "),
  "VersionNumber"    : str(sys.version_info[0]) + "." + str(sys.version_info[1]),
  "Prefix"           : getattr(sys, "prefix", ""),
  "ExecPrefix"       : getattr(sys, "exec_prefix", ""),
  "BaseExecPrefix"   : getattr(sys, "base_exec_prefix", ""),
  "PythonPath"       : pathsep.join((x or "." for x in sys.path)),
  "LIBPL"            : sysconfig.get_config_var("LIBPL"),
  "LIBDIR"           : sysconfig.get_config_var("LIBDIR"),
  "SharedLibrary"    : sysconfig.get_config_var("Py_ENABLE_SHARED"),
  "Executable"       : getattr(sys, "executable", ""),
  "BaseExecutable"   : getattr(sys, "_base_executable", ""),
}

# detect if this is a conda managed python
# https://stackoverflow.com/a/21282816/5128728
if sys.version_info >= (3, 7):
  is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
else:
  is_conda = 'conda' in sys.version
config['IsConda'] = is_conda

# Read numpy configuration (if available)
try:
  import numpy
  config["NumpyPath"]    = str(numpy.__path__[0])
  config["NumpyVersion"] = str(numpy.__version__)
except:
  pass

# Read required module information (if requested)
try:
  required_module = os.environ["RETICULATE_REQUIRED_MODULE"]
  if required_module is not None and len(required_module) > 0:
    config["RequiredModule"] = required_module
    config["RequiredModulePath"] = module_path(required_module)
except:
  pass

# Write configuration to stdout
lines = [str(key) + ": " + str(val) for (key, val) in config.items()]
text = "\n".join(lines)
sys.stdout.write(text)
