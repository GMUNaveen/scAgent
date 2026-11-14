#!/usr/bin/env bash
# Download and unpack the Tabula Muris data into data/.
# Usage: scripts/get_data.sh [URL]   (or set DATA_URL)
# Default is the GitHub Release asset attached to this repo (see README "Data").
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${1:-${DATA_URL:-https://github.com/GMUNaveen/scAgent/releases/download/data-v1/scrna_data.tar.gz}}"
if [ -d data/droplet ]; then echo "data/droplet already exists; nothing to do."; exit 0; fi
echo "Downloading $URL"
curl -L --fail -o scrna_data.tar.gz "$URL"
tar -xzf scrna_data.tar.gz
rm scrna_data.tar.gz
echo "Done: $(ls data/droplet | wc -l) samples in data/droplet"
