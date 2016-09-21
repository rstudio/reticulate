
## TensorFlow for R

[TensorFlow™](https://tensorflow.org) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 

The **tensorflow** R package provides access to the complete TensorFlow API from within R.

## Installation

1. Follow the instructions here to install the main TensorFlow distribution:

  https://www.tensorflow.org/get_started/os_setup.html#download-and-setup

2. Clone and install the **tensorflow** R package:

    ```bash
    git clone git@github.com:jjallaire/tensorflow.git
    R CMD build tensorflow && R CMD INSTALL tensorflow_0.2.0.tar.gz
    ```

## Documentation

You can access documentation for the **tensorflow** package as follows:

#### Ubuntu

```bash
xdg-open tensorflow/docs/index.html
```

#### Mac OS X

```bash
open tensorflow/docs/index.html
```




