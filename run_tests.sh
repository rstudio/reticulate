#!/bin/bash

Rscript -e 'Sys.setenv(TENSORFLOW_TEST_EXAMPLES="1");devtools::check("tensorflow", document = FALSE)'
