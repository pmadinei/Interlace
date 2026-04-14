#!/bin/bash
# INTERLACE training script
# Usage: bash scripts/train_interlace.sh

set -e

# =============================================================================
# Configuration (modify these for your setup)
# =============================================================================

# Model
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"   # or Qwen/Qwen3-VL-4B-Instruct

# Pruning
PRUNING_STRATEGY="interlace"               # interlace, interlace_oa, interlace_tn, random, consecutive
DROP_RATIO=0.25                            # 0.10, 0.15, 0.20, 0.25

# Hidden state similarity paths (required for similarity-based strategies)
HS_DIR="./hidden_states"
MODEL_SHORT=$(echo ${MODEL_NAME} | tr '/' '-' | sed 's/.*-//')
HS_PACK1="${HS_DIR}/${MODEL_SHORT}_pack1_hidden_state_similarities.json"
HS_PACK2="${HS_DIR}/${MODEL_SHORT}_pack2_hidden_state_similarities.json"
HS_PACK3="${HS_DIR}/${MODEL_SHORT}_pack3_hidden_state_similarities.json"

# Dataset
DATASET_PATH="/path/to/FineVision_01.json"   # Set to your dataset path

# Training hyperparameters
LR=1e-5
BATCH_SIZE=16
GRAD_ACCUM_STEPS=2
NUM_EPOCHS=1

# Output
DROP_PCT=$(echo "${DROP_RATIO} * 100" | bc | cut -d. -f1)
RUN_NAME="interlace_${MODEL_SHORT}_${DROP_PCT}pc"
OUTPUT_DIR="./checkpoints/${RUN_NAME}"

# DeepSpeed
DEEPSPEED_CONFIG="./configs/zero3.json"

# Distributed training
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NUM_GPUS=${NUM_GPUS:-1}

# =============================================================================
# Launch training
# =============================================================================

torchrun --nproc_per_node=${NUM_GPUS} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
    src/train/train_interlace.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path "${MODEL_NAME}" \
    --dataset_use "${DATASET_PATH}" \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate ${LR} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --report_to wandb \
    --pruning_strategy ${PRUNING_STRATEGY} \
    --drop_ratio ${DROP_RATIO} \
    --hs_pack1_path "${HS_PACK1}" \
    --hs_pack2_path "${HS_PACK2}" \
    --hs_pack3_path "${HS_PACK3}"
