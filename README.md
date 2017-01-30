[![Travis-CI Build Status](https://travis-ci.org/rstudio/tensorflow.svg?branch=master)](https://travis-ci.org/rstudio/tensorflow)

## TensorFlow for R

[TensorFlow™](https://tensorflow.org) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 

The [TensorFlow API](https://www.tensorflow.org/api_docs/python/index.html) is composed of a set of Python modules that enable constructing and executing TensorFlow graphs. The tensorflow package provides access to the complete TensorFlow API from within R. 

## Installing TensorFlow

You can install the main TensorFlow distribution from here:

<https://www.tensorflow.org/get_started/os_setup.html#download-and-setup>

Some important notes on compatibility:

* You should NOT install TensorFlow with Anaconda as there [are issues with](https://github.com/ContinuumIO/anaconda-issues/issues/498) the way Anaconda builds the python shared library that prevent dynamic linking from R.

* If you install TensorFlow within a Virtualenv environment you'll need to be sure to use that same environment when loading the tensorflow R package (see below for details).

## Installing the R Package

You can install the tensorflow R package from GitHub as follows:

```r
devtools::install_github("rstudio/tensorflow")
```

Note that the tensorflow package includes native C/C++ code so it's installation requires [R Tools](https://cran.r-project.org/bin/windows/Rtools/) on Windows and [Command Line Tools](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/) on OS X. If the package installation fails because of inability to compile then install the appropriate tools for your platform based on the links above and try again.

## Binding to Tensorflow

When it is loaded the tensorflow R package will scan your system for various versions of Python and attempt to identify one which includes a compatible version of tensorflow to bind to. If tensorflow is not automatically located in this fashion you should do one of the following:

* Ensure that the version of python where you installed tensorflow is the default python on the `PATH` within your R session.

* Set the `TENSORFLOW_PYTHON` environment variable to the full path of the python binary before loading the package, for example:

    ```r
    Sys.setenv(TENSORFLOW_PYTHON="/usr/local/bin/python")
    library(tensorflow)
    ```

## Verifying Installation

You can verify that your installation is working correctly by running this script:

```r
library(tensorflow)
sess = tf$Session()
hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)
```

## Documentation

We recommend that users try out TensorFlow's high-level [TF.Learn module](https://arxiv.org/abs/1612.04251) which requires  less use of lower-level TensorFlow APIs. Some basic examples can be found [here](https://github.com/rstudio/tensorflow/tree/master/inst/examples/tflearn).

See the package website for additional details on using the TensorFlow API from R: <https://rstudio.github.io/tensorflow>

See the TensorFlow API reference for details on all of the modules, classes, and functions within the API: <https://www.tensorflow.org/api_docs/python/index.html>

The tensorflow package provides code completion and inline help for the TensorFlow API when running within the RStudio IDE. In order to take advantage of these features you should also install the current [Preview Release](https://www.rstudio.com/products/rstudio/download/preview/) of RStudio.




