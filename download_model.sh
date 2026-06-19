#!/bin/bash
# Downloads the trained Pashto TTS model from GitHub Releases.
# Run: bash download_model.sh

set -e

REPO="HussainM10/pashto-tts"
FILE="G_314000.pth"
DEST="model/$FILE"

mkdir -p model

if [ -f "$DEST" ]; then
    echo "Model already exists at $DEST — skipping download."
    exit 0
fi

echo "Downloading $FILE (458 MB)..."
curl -L \
    "https://github.com/$REPO/releases/latest/download/$FILE" \
    -o "$DEST" \
    --progress-bar

echo "Done. Model saved to $DEST"
echo "You can now run: python app.py"
