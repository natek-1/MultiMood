#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define dataset directory
MELD_DIR="meld_dataset"

echo "Creating directory $MELD_DIR..."
mkdir -p "$MELD_DIR"
cd "$MELD_DIR"

# Download the main dataset file
echo "Downloading the MELD dataset..."
wget --no-check-certificate -O MELD.Raw.tar.gz https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# Extract the main archive
echo "Extracting main archive..."
# Extract everything into a temporary folder to avoid clutter and handle unpredictable tar structures
mkdir -p tmp_extraction
tar -xf MELD.Raw.tar.gz -C tmp_extraction/

# Move the required files into the corresponding split folders
echo "Organizing files..."
for split in train dev test; do
    echo "Processing $split..."
    mkdir -p "$split"
    
    # Recursively find and move the CSV file for this split
    find tmp_extraction -type f -name "${split}_sent_emo.csv" -exec mv {} "$split/" \;
    
    # Recursively find the tar.gz or tar file for this split
    TAR_FILE=$(find tmp_extraction -type f \( -name "${split}.tar.gz" -o -name "${split}.tar" \) | head -n 1)
    
    if [ -n "$TAR_FILE" ]; then
        echo "Extracting $TAR_FILE into $split..."
        tar -xf "$TAR_FILE" -C "$split/"
    else
        echo "Warning: Compressed file for $split not found!"
    fi
done

# Cleanup
echo "Cleaning up compressed and temporary files..."
rm -f MELD.Raw.tar.gz
rm -rf tmp_extraction

echo "MELD dataset successfully downloaded and extracted into $MELD_DIR."
