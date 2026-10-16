#!/bin/bash
set -euo pipefail

INPUT_ROOT="/mnt/data/hapq/TRAINING_DATA/FlyingThings3D"
OUTPUT_ROOT="/mnt/data/hapq/TRAINING_DATA/FlyingThings3D_extra"
SCRIPT="/mnt/data/hapq/mvinverse/inference.py"
for pass in frames_cleanpass frames_finalpass; do
    for category in A B C; do
        echo "Processing FlyingThings3D $pass/$category"
        while IFS= read -r -d '' dir; do
            rel="${dir#"$INPUT_ROOT/$pass/TRAIN/$category/"}"
            out_dir="$OUTPUT_ROOT/$pass/TRAIN/$category/$rel"
            mkdir -p "$out_dir"
            echo "  $rel"
            python "$SCRIPT" --data_path "$dir" --save_path "$out_dir" --chunk_size 10
        done < <(find "$INPUT_ROOT/$pass/TRAIN/$category" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)
    done
done

echo "FlyingThings3D albedo generation completed."
