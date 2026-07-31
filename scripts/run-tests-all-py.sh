#!/usr/bin/env bash
set -euo pipefail

versions=(
  3.14
  3.14t
  3.13
  3.12
  3.11
  3.10
  3.9
)

export COLUMNS=80
export LINES=24

for version in "${versions[@]}"; do
  echo "=== Running reticulate tests for Python ${version} ==="
  RETICULATE_TEST_PYTHON_VERSION="${version}" \
    Rscript -e 'devtools::test()'
done
