
## TensorFlow for R

[TensorFlow™](https://tensorflow.org) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 

TensorFlow is written in C++ and Python, and is most typically used via it's [Python API](https://www.tensorflow.org/api_docs/python/index.html). The tensorflow R package provides access to the complete TensorFlow Python API from within R.

## Installation

1. Install the main TensorFlow distribution:

  https://www.tensorflow.org/get_started/os_setup.html#download-and-setup

2. Install the tensorflow R package:

    ```r
    library(devtools)
    install_github("rstudio/tensorflow", auth_token = "b3ed53b2a3f239d1a994ee7193139b4a79daaf8c")
    ```

    The tensorflow package will be built against the default version of python found in the `PATH`. If you want to build against a specific version of python you can define the `TENSORFLOW_PYTHON_VERSION` environment variable before installing. For example:

    ```r
    Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
    install_github("rstudio/tensorflow", auth_token = "b3ed53b2a3f239d1a994ee7193139b4a79daaf8c")
    ```

You can verify that your installation is working correctly by running this script:

```r
library(tensorflow)
sess = tf$Session()
hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)
```

## Documentation

The documentation for the tensorflow package will be published using gh-pages once this repository is public. For the time being you can access a copy of the documentation installed along with the package by calling the `view_docs` function: 

```r
tensorflow:::view_docs()
```

### RStudio IDE

The current [Preview Release](https://www.rstudio.com/products/rstudio/download/preview/) of RStudio includes enhancements that the tensorflow package makes use of to provide improved completion and inline help for the TensorFlow API. If you are using RStudio it's therefore highly recommended that you install the preview release.




