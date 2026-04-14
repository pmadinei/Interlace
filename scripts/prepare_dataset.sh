#!/bin/bash
# Prepare FineVision dataset for INTERLACE fine-tuning.

set -e

OUTPUT_DIR="${1:-./data}"
SAMPLE_PORTION="${2:-0.01}"  # 1% of FineVision

echo "Preparing FineVision dataset (${SAMPLE_PORTION} portion)"
echo "Output directory: ${OUTPUT_DIR}"

python src/dataset_prep.py \
    --sample_portion ${SAMPLE_PORTION} \
    --output_dir "${OUTPUT_DIR}" \
    --sample_counts_path dataset_sample_counts.json \
    --seed 42

echo "Done! Dataset saved to ${OUTPUT_DIR}/"
