#!/usr/bin/env bash
# Tokenizes an OMOP MEDS dataset and trains an ETHOS model on it.
#
# Usage:
#   export input_dir=/path/to/omop_meds/data
#   export output_dir=/path/to/output
#   ./scripts/run_omop_pipeline.sh
#
# Override any default below by exporting the same-named variable before running,
# e.g. `export DATASET=omop_no_lab` to skip lab/LOINC data.

set -e

: "${input_dir:?input_dir must be set, e.g. export input_dir=/path/to/meds/data}"
: "${output_dir:?output_dir must be set, e.g. export output_dir=/path/to/output}"

# --- tokenization ---
DATASET=${DATASET:-omop}             # or omop_no_lab to skip LOINC/lab data
OUT_FN=${OUT_FN:-all}
NUM_WORKERS=${NUM_WORKERS:-$(nproc)}

# --- model / training ---
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}
VAL_SIZE=${VAL_SIZE:-6}              # fraction in (0,1), or millions of tokens if >=1
BATCH_SIZE=${BATCH_SIZE:-32}
N_POSITIONS=${N_POSITIONS:-2048}
N_LAYER=${N_LAYER:-6}
N_HEAD=${N_HEAD:-12}
N_EMBD=${N_EMBD:-768}
DROPOUT=${DROPOUT:-0.3}
LR=${LR:-0.0006}
MIN_LR=${MIN_LR:-0.00001}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
MAX_ITERS=${MAX_ITERS:-200000}
WARMUP_ITERS=${WARMUP_ITERS:-5000}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-100000}

data_path="${output_dir%/}/${OUT_FN}"
model_name="layer_${N_LAYER}_do_${DROPOUT}"

echo "=== Tokenizing '$input_dir' (dataset=$DATASET) with $NUM_WORKERS worker(s) ==="
ethos_tokenize -m worker="range(0,${NUM_WORKERS})" \
    dataset="$DATASET" \
    input_dir="$input_dir" \
    output_dir="$output_dir" \
    out_fn="$OUT_FN"

echo "=== Training on $NUM_GPUS GPU(s), data_fp=$data_path ==="
torchrun --no_python --standalone --nproc_per_node="$NUM_GPUS" ethos_train \
    data_fp="$data_path" \
    val_size="$VAL_SIZE" \
    batch_size="$BATCH_SIZE" \
    n_positions="$N_POSITIONS" \
    n_layer="$N_LAYER" \
    n_head="$N_HEAD" \
    n_embd="$N_EMBD" \
    dropout="$DROPOUT" \
    lr="$LR" \
    min_lr="$MIN_LR" \
    log_interval=10 \
    eval_interval=1500 \
    gradient_accumulation_steps="$GRAD_ACCUM_STEPS" \
    warmup_iters="$WARMUP_ITERS" \
    max_iters="$MAX_ITERS" \
    lr_decay_iters="$LR_DECAY_ITERS" \
    out_dir="${data_path}/models/${model_name}"
