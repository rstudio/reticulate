#!/bin/bash
set -e

# Python dependencies
pip install --upgrade pip --user
pip install numpy --user

# tensorflow for separate os
if [ ${TRAVIS_OS_NAME} == "linux" ]; then
  pip install --user https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
fi
if [ ${TRAVIS_OS_NAME} == "osx" ]; then
  pip install --user https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.12.1-py2-none-any.whl
fi
