
import sys
import os
import sysconfig
import platform
import imp

sys.stdout.write('Version: ' + str(sys.version).replace('\n', ' '))
sys.stdout.write('\nVersionNumber: ' + str(sys.version_info.major) + '.' + str(sys.version_info.minor))
if not platform.system() == 'Windows':
  sys.stdout.write('\nLIBPL: ' + sysconfig.get_config_vars('LIBPL')[0])
  sys.stdout.write('\nLIBDIR: ' + sysconfig.get_config_vars('LIBDIR')[0])
sys.stdout.write('\nPREFIX: ' + sysconfig.get_config_vars('prefix')[0])
sys.stdout.write('\nEXEC_PREFIX: ' + sysconfig.get_config_vars('exec_prefix')[0])
sys.stdout.write("\nArchitecture: "  + platform.architecture()[0])

try:
  import numpy
  sys.stdout.write('\nNumpyPath: ' + str(numpy.__path__[0]))
  sys.stdout.write('\nNumpyVersion: ' + str(numpy.__version__))
except Exception:
  pass


if "RPY_REQUIRED_MODULE" in os.environ:
  required_module = os.environ.get("RPY_REQUIRED_MODULE")
  try:
    sys.stdout.write('\nRequiredModule: ' + required_module)
    sys.stdout.write('\nRequiredModulePath: ' + str(imp.find_module(required_module)[1]))
  except Exception:
    pass


