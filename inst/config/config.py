
import sys
import os
import platform

# The 'imp' module is deprecated since Python 3.4, and the use of
# 'importlib' is recommended instead.
if sys.version < '3.4':
  import imp
  def module_path(name):
    spec = imp.find_module(name)
    return spec[1]
else:
  from importlib import util
  def module_path(name):
    spec = util.find_spec(name)
    origin = spec.origin
    return origin[:origin.rfind('/')]

sys.stdout.write('Version: ' + str(sys.version).replace('\n', ' '))
sys.stdout.write('\nVersionNumber: ' + str(sys.version_info[0]) + '.' + str(sys.version_info[1]))

try:
  import sysconfig
  if not platform.system() == 'Windows':
    sys.stdout.write('\nLIBPL: ' + sysconfig.get_config_vars('LIBPL')[0])
    sys.stdout.write('\nLIBDIR: ' + sysconfig.get_config_vars('LIBDIR')[0])
  sys.stdout.write('\nPREFIX: ' + sysconfig.get_config_vars('prefix')[0])
  sys.stdout.write('\nEXEC_PREFIX: ' + sysconfig.get_config_vars('exec_prefix')[0])
except Exception:
  pass

sys.stdout.write("\nArchitecture: "  + platform.architecture()[0])

try:
  import numpy
  sys.stdout.write('\nNumpyPath: ' + str(numpy.__path__[0]))
  sys.stdout.write('\nNumpyVersion: ' + str(numpy.__version__))
except Exception:
  pass


if "RETICULATE_REQUIRED_MODULE" in os.environ:
  required_module = os.environ.get("RETICULATE_REQUIRED_MODULE")
  try:
    sys.stdout.write('\nRequiredModule: ' + required_module)
    sys.stdout.write('\nRequiredModulePath: ' + str(module_path(required_module)))
  except Exception:
    pass


