#!/usr/bin/env bash
set -euo pipefail

ZIP_NAME="model.zip"
rm -f "$ZIP_NAME"

mkdir -p models

echo "Bundling submission files into $ZIP_NAME..."
zip -r "$ZIP_NAME" solution.py requirements.txt models/ -x "*.DS_Store" "*__pycache__*"

echo "----------------------------------------"
echo "Created $ZIP_NAME"
echo "Next step on VM: sudo check-submission $ZIP_NAME"
