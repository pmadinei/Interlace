#!/bin/bash
# Compute hidden state similarities for INTERLACE layer selection.
# Must be run for pack_size=1, pack_size=2, and pack_size=3.

set -e

MODEL_PATH="${1:-Qwen/Qwen3-VL-8B-Instruct}"
DATASET_PATH="${2:-/path/to/FineVision_01.json}"
OUTPUT_DIR="${3:-./hidden_states}"
SAMPLE_PORTION="${4:-0.1}"

echo "Computing hidden state similarities for ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"

for PACK_SIZE in 1 2 3; do
    echo "--- Pack size: ${PACK_SIZE} ---"
    python src/get_hidden_states.py \
        --model_path "${MODEL_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --sample_portion ${SAMPLE_PORTION} \
        --pack_size ${PACK_SIZE} \
        --random_seed 42
done

echo "Done! Similarity scores saved to ${OUTPUT_DIR}/"
