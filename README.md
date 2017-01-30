[![Travis-CI Build Status](https://travis-ci.org/rstudio/tensorflow.svg?branch=master)](https://travis-ci.org/rstudio/tensorflow)

## TensorFlow for R

[TensorFlow™](https://tensorflow.org) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 

The [TensorFlow API](https://www.tensorflow.org/api_docs/python/index.html) is composed of a set of Python modules that enable constructing and executing TensorFlow graphs. The tensorflow package provides access to the complete TensorFlow API from within R. 

## Installation

First, install the main TensorFlow distribution from here:

<https://www.tensorflow.org/get_started/os_setup.html#download-and-setup>

If you install TensorFlow within a virtualenv environment you'll need to be sure to use that same environment when loading the tensorflow R package (see below for details on how to do this).

Next, install the tensorflow R package from GitHub as follows:

```r
devtools::install_github("rstudio/tensorflow")
```

Note that the tensorflow package includes native C/C++ code so it's installation requires [R Tools](https://cran.r-project.org/bin/windows/Rtools/) on Windows and [Command Line Tools](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/) on OS X. If the package installation fails because of inability to compile then install the appropriate tools for your platform based on the links above and try again.

#### Locating TensorFlow

When it is loaded the tensorflow R package scans the system for the version of python where TensorFlow is installed. If automatic detection doesn't work or if you want to exercise more control over which version(s) of python and TensorFlow are used you can specify an explicit `TENSORFLOW_PYTHON` environment variable to force probing for TensorFlow within a specific version of python, for example:

```r
Sys.setenv(TENSORFLOW_PYTHON="/usr/local/bin/python")
library(tensorflow)
```

The tensorflow package will look in this location first, then look for python on the system `PATH`, then scan additional locations where python is conventionally installed (e.g. `/usr/local/bin`, `/opt/python/bin`, etc.).
## Verifying Installation

You can verify that your installation is working correctly by running this script:

```r
library(tensorflow)
sess = tf$Session()
hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)
```

## Documentation

See the package website for additional details on using the TensorFlow API from R: <https://rstudio.github.io/tensorflow>

See the TensorFlow API reference for details on all of the modules, classes, and functions within the API: <https://www.tensorflow.org/api_docs/python/index.html>

The tensorflow package provides code completion and inline help for the TensorFlow API when running within the RStudio IDE. In order to take advantage of these features you should also install the current [Preview Release](https://www.rstudio.com/products/rstudio/download/preview/) of RStudio.




