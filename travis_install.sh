#!/bin/bash

# create virtualenv
deactivate
virtualenv --system-site-packages testenv
source testenv/bin/activate

# Python dependencies
sudo pip install --upgrade pip
sudo pip install numpy
# tensorflow for separate os
if [ ${TRAVIS_OS_NAME} == "linux" ]; then
  sudo pip install https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
fi
if [ ${TRAVIS_OS_NAME} == "osx" ]; then
  sudo pip install https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac1-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.10.0-py2-none-any.whl
fi

# R dependencies
curl -OL http://raw.github.com/craigcitro/r-travis/master/scripts/travis-tool.sh
chmod 755 ./travis-tool.sh
./travis-tool.sh bootstrap
./travis-tool.sh install_aptget r-cran-testthat r-cran-devtools r-cran-rcpp
./travis-tool.sh install_deps

# package
git clone git@github.com:rstudio/tensorflow.git
