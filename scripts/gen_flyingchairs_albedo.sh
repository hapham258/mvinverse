#!/bin/bash
set -e

DATA_ROOT="/mnt/data/hapq/TRAINING_DATA/FlyingChairs/FlyingChairs_release/data"
OUTPUT_ROOT="/mnt/data/hapq/TRAINING_DATA/FlyingChairs_extra"
SCRIPT="/mnt/data/hapq/mvinverse/inference.py"

for suffix in img1 img2; do
    image_dir=$(mktemp -d)
    trap 'rm -rf "$image_dir"' EXIT

    while IFS= read -r file; do
        ln -sf "$DATA_ROOT/$file" "$image_dir/$file"
    done < <(find "$DATA_ROOT" -maxdepth 1 -type f -name "*_${suffix}.ppm" -printf '%f\n' | sort)

    save_dir="$OUTPUT_ROOT/FlyingChairs_release/data"
    mkdir -p "$save_dir"
    echo "Processing FlyingChairs $suffix"
    python "$SCRIPT" --data_path "$image_dir" --save_path "$save_dir" --chunk_size 2 --device cuda
    rm -rf "$image_dir"
    trap - EXIT
done

echo "All FlyingChairs images processed."
