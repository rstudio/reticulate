
## TensorFlow for R

[TensorFlow™](https://tensorflow.org) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 

The [TensorFlow API](https://www.tensorflow.org/api_docs/python/index.html) is composed of a set of Python modules that enable constructing and executing TensorFlow graphs. The tensorflow package provides access to the complete TensorFlow API from within R. 

## Installation

1. Install the main TensorFlow distribution:

  https://www.tensorflow.org/get_started/os_setup.html#download-and-setup

2. Install the tensorflow R package:

    ```r
    library(devtools)
    install_github("rstudio/tensorflow")
    ```

    The tensorflow package will be built against the default version of python found in the `PATH`. If you want to build against a specific version of python you can define the `TENSORFLOW_PYTHON_VERSION` environment variable before installing. For example:

    ```r
    Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
    install_github("rstudio/tensorflow")
    ```

3. Verify that your installation is working correctly by running this script:

    ```r
    library(tensorflow)
    sess = tf$Session()
    hello <- tf$constant('Hello, TensorFlow!')
    sess$run(hello)
    ```

The tensorflow package provides code completion and inline help for the TensorFlow API when running within the RStudio IDE. In order to take advantage of these features you should also install the current [Preview Release](https://www.rstudio.com/products/rstudio/download/preview/) of RStudio.


## Documentation

See the package website for additional details on using the TensorFlow API from R: <https://rstudio.github.io/tensorflow>

See the TensorFlow API reference for details on all of the modules, classes, and functions within the API: <https://www.tensorflow.org/api_docs/python/index.html>





