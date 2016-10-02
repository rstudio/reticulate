#!/bin/bash

Rscript -e 'Sys.setenv(TENSORFLOW_TEST_EXAMPLES="1");devtools::install();testthat::test_package("tensorflow");'
