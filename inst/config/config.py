
import sys
import sysconfig
import platform
import pkgutil

sys.stdout.write('Version: ' + str(sys.version).replace('\n', ' '))
sys.stdout.write('\nVersionNumber: ' + str(sys.version_info.major) + '.' + str(sys.version_info.minor))
if not platform.system() == 'Windows':
  sys.stdout.write('\nLIBPL: ' + sysconfig.get_config_vars('LIBPL')[0])
  sys.stdout.write('\nLIBDIR: ' + sysconfig.get_config_vars('LIBDIR')[0])
sys.stdout.write('\nPREFIX: ' + sysconfig.get_config_vars('prefix')[0])
sys.stdout.write('\nEXEC_PREFIX: ' + sysconfig.get_config_vars('exec_prefix')[0])

try:
  import numpy
  sys.stdout.write('\nNumpyPath: ' + str(numpy.__path__[0]))
  sys.stdout.write('\nNumpyVersion: ' + str(numpy.__version__))
except Exception, e:
  pass

try:
  import tensorflow
  sys.stdout.write('\nTensorflowPath: ' + str(tensorflow.__path__[0]))
  sys.stdout.write('\nTensorflowVersion: ' + str(tensorflow.__version__))
except Exception, e:
  pass
