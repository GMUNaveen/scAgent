#!/usr/bin/env bash
# Package data/ into one compressed archive for upload (e.g. as a GitHub Release asset).
set -euo pipefail
cd "$(dirname "$0")/.."
tar -czf scrna_data.tar.gz data/
shasum -a 256 scrna_data.tar.gz | tee scrna_data.tar.gz.sha256
ls -lh scrna_data.tar.gz
