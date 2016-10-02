#!/bin/bash

Rscript -e 'Sys.setenv(TENSORFLOW_PYTHON=system("which python", intern = T));library(tensorflow)'