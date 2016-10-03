#!/bin/bash

Rscript -e 'Sys.setenv(TENSORFLOW_TEST_EXAMPLES="1");Sys.setenv(R_GCTORTURE="25");devtools::install();testthat::test_package("tensorflow");'
