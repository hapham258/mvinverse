#!/bin/bash
set -e

DATA_ROOT="/mnt/data/hapq/TRAINING_DATA/TartanAir"
OUTPUT_ROOT="/mnt/data/hapq/TRAINING_DATA/TartanAir_extra"
SCRIPT="/mnt/data/hapq/mvinverse/inference.py"

for scene in "$DATA_ROOT"/*/; do
    for difficulty in "$scene"*/; do
        for seq in "$difficulty"*/; do
            image_dir="${seq}image_left"
            [ -d "$image_dir" ] || continue

            scene_name=$(basename "$(dirname "$(dirname "$seq")")")
            difficulty_name=$(basename "$(dirname "$seq")")
            seq_name=$(basename "$seq")
            save_dir="$OUTPUT_ROOT/$scene_name/$difficulty_name/$seq_name/albedo_left"

            mkdir -p "$save_dir"
            echo "Processing $scene_name/$difficulty_name/$seq_name"
            python "$SCRIPT" --data_path "$image_dir" --save_path "$save_dir"
        done
    done
done

echo "All sequences processed."
