#!/bin/bash

# Base directory containing the dataset
BASE_DIR="measurements/data/dataset"

# Subdirectories to process
SUBDIRS=("train/images" "train/masks" "val/images" "val/masks" "test/images" "test/masks")

# Percentage of files to commit at a time
PERCENTAGE=5

# Collect all files from the specified subdirectories
ALL_FILES=()
for SUBDIR in "${SUBDIRS[@]}"; do
    FULL_DIR="$BASE_DIR/$SUBDIR"
    
    # Check if the subdirectory exists
    if [ ! -d "$FULL_DIR" ]; then
        echo "Directory $FULL_DIR does not exist, skipping."
        continue
    fi
    
    # Add files to the list
    FILES=($(find "$FULL_DIR" -type f))
    ALL_FILES+=("${FILES[@]}")
done

# Calculate the number of files to commit per batch
TOTAL_FILES=${#ALL_FILES[@]}
FILES_TO_COMMIT=$((TOTAL_FILES * PERCENTAGE / 100))

if [ $FILES_TO_COMMIT -eq 0 ]; then
    echo "No files to commit."
    exit 1
fi

# Shuffle the list of all files (replace `shuf` with a manual shuffle using `sort` and `awk`)
shuffled_files=($(printf "%s\n" "${ALL_FILES[@]}" | sort -R))

# Function to commit and push a batch of files
commit_and_push() {
    local start=$1
    local end=$2
    
    for ((i=start; i<end; i++)); do
        git add "${shuffled_files[$i]}"
    done
    
    git commit -m "Committed files ${start} to ${end} (${PERCENTAGE}% of the dataset)"
    git push origin main  # Change 'main' to your branch name if different
}

# Commit and push files in batches
for ((i=0; i<TOTAL_FILES; i+=FILES_TO_COMMIT)); do
    end=$((i + FILES_TO_COMMIT))
    if [ $end -gt $TOTAL_FILES ]; then
        end=$TOTAL_FILES
    fi
    commit_and_push $i $end
    echo "Committed and pushed files $i to $end."
done

echo "All files committed and pushed."