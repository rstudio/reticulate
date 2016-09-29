
## TensorFlow for R

[TensorFlow™](https://tensorflow.org) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 

The [TensorFlow API](https://www.tensorflow.org/api_docs/python/index.html) is composed of a set of Python modules that enable constructing and executing TensorFlow graphs. The tensorflow package provides access to the complete TensorFlow API from within R. 

## Installing TensorFlow

You can install the main TensorFlow distribution from here:

<https://www.tensorflow.org/get_started/os_setup.html#download-and-setup>

Note that if you install TensorFlow within a Virtualenv or Conda environment you'll need to be sure to use that same environment when installing the tensorflow R package (see below for details).

## Installing the R Package

If you installed TensorFlow via pip with your system default version of python then you can install the tensorflow R package as follows:

```r
devtools::install_github("rstudio/tensorflow")
```

If you are using a different version of python for TensorFlow, you should set the `TENSORFLOW_PYTHON` environment variable to the full path of the python binary before installing, for example:

```r
Sys.setenv(TENSORFLOW_PYTHON="~/anaconda/envs/tensorflow/bin/python")
devtools::install_github("rstudio/tensorflow")
```

Or:

```r
Sys.setenv(TENSORFLOW_PYTHON="/usr/local/bin/python")
devtools::install_github("rstudio/tensorflow")
```

If you only need to customize the version of python used (for example specifing python 3 on an Ubuntu system), you can set the `TENSORFLOW_PYTHON_VERSION` environment variable before installation:

```r
Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
devtools::install_github("rstudio/tensorflow")
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

See the package website for additional details on using the TensorFlow API from R: <https://rstudio.github.io/tensorflow>

See the TensorFlow API reference for details on all of the modules, classes, and functions within the API: <https://www.tensorflow.org/api_docs/python/index.html>

The tensorflow package provides code completion and inline help for the TensorFlow API when running within the RStudio IDE. In order to take advantage of these features you should also install the current [Preview Release](https://www.rstudio.com/products/rstudio/download/preview/) of RStudio.




