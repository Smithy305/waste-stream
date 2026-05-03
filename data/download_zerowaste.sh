#!/usr/bin/env bash
# Download the ZeroWaste dataset from Zenodo.
# Usage: bash data/download_zerowaste.sh [TARGET_DIR]
#
# The dataset is ~2.5 GB. If the automatic download fails, grab it manually:
#   https://zenodo.org/records/6412647

set -euo pipefail

TARGET_DIR="${1:-data/zerowaste}"
ZENODO_URL="https://zenodo.org/records/6412647/files/zerowaste-f-final.zip"
ZIP_FILE="${TARGET_DIR}/zerowaste-f-final.zip"

mkdir -p "$TARGET_DIR"

if [ -d "${TARGET_DIR}/train" ] && [ -d "${TARGET_DIR}/val" ]; then
    echo "ZeroWaste dataset already exists at ${TARGET_DIR}. Skipping download."
    exit 0
fi

echo "Downloading ZeroWaste dataset to ${TARGET_DIR}..."
if command -v wget &> /dev/null; then
    wget -c "$ZENODO_URL" -O "$ZIP_FILE"
elif command -v curl &> /dev/null; then
    curl -L -C - -o "$ZIP_FILE" "$ZENODO_URL"
else
    echo "Error: neither wget nor curl found. Install one or download manually:"
    echo "  $ZENODO_URL"
    exit 1
fi

echo "Extracting..."
unzip -q -o "$ZIP_FILE" -d "$TARGET_DIR"
rm -f "$ZIP_FILE"

echo "Done. Dataset extracted to ${TARGET_DIR}."
