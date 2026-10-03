#!/usr/bin/env bash
# Linear probing of a pretrained ETHOS checkpoint on a set of ACES/MEDS label cohorts.
#
# Each cohort is a directory of label parquet files (subject_id, prediction_time, boolean_value),
# either directly in it or in a `data/` subfolder. The label table carries no train/test column;
# the split is by patient: the same labels are matched against each tokenized MEDS split, and a
# split only keeps the labels of patients it contains. So the pipeline's `train` and `tuning`
# splits give the training features and `held_out` gives the test features.
#
# Prerequisites:
#   - all three MEDS splits tokenized with the SAME vocab (the pipeline script only does train
#     and tuning; held_out needs `ethos_tokenize ... input_dir=<meds>/held_out vocab=<train>
#     out_fn=held_out`)
#   - `pip install pandas scikit-learn` (used by train_logreg.py)
#
# Usage:
#   export MODEL_FP=/path/to/best_model.pt
#   export COHORTS_DIR=~/ohdsi_cumc_deid_2023q4r3_cleaned_corrected/cehrgpt_tasks_local_time
#   export TOKENIZED_DIR=/path/to/ethos-output     # contains train/ tuning/ held_out/
#   export OUT_DIR=/path/to/linear_probing
#   ./scripts/linear_prob/run_cohorts.sh                 # all cohorts
#   ./scripts/linear_prob/run_cohorts.sh hf_readmission_meds t2dm_hf
#
# Finished steps are skipped, so after an interruption just run it again. Run it under tmux or
# `setsid nohup ... &`, not plain nohup.
#
# Optional: TRAIN_SPLITS="train tuning", TEST_SPLIT=held_out, EXCLUDE="logs", NUM_GPUS (cohorts
# are spread over the GPUs, one worker per GPU), AVERAGE_OVER_SEQUENCE=false (true: mean-pool
# hidden states instead of using the last token).

set -u -o pipefail

: "${MODEL_FP:?MODEL_FP must point to the checkpoint, e.g. .../models/<name>/best_model.pt}"
: "${COHORTS_DIR:?COHORTS_DIR must be set}"
: "${TOKENIZED_DIR:?TOKENIZED_DIR must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"

TRAIN_SPLITS=${TRAIN_SPLITS:-"train tuning"}
TEST_SPLIT=${TEST_SPLIT:-held_out}
EXCLUDE=${EXCLUDE:-logs}
AVERAGE_OVER_SEQUENCE=${AVERAGE_OVER_SEQUENCE:-false}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}
DEVICE=cuda
if [[ "$NUM_GPUS" -eq 0 ]]; then
    NUM_GPUS=1
    DEVICE=cpu
fi

COHORTS_DIR=${COHORTS_DIR%/}
TOKENIZED_DIR=${TOKENIZED_DIR%/}
OUT_DIR=${OUT_DIR%/}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

[[ -f "$MODEL_FP" ]] || { echo "Checkpoint not found: $MODEL_FP" >&2; exit 1; }
for split in $TRAIN_SPLITS $TEST_SPLIT; do
    if ! compgen -G "$TOKENIZED_DIR/$split/[0-9]*.safetensors" >/dev/null; then
        echo "No tokenized shards in '$TOKENIZED_DIR/$split'. Tokenize that split first, reusing" \
             "the train vocab (vocab=$TOKENIZED_DIR/train)." >&2
        exit 1
    fi
done

if [[ $# -gt 0 ]]; then
    cohorts=("$@")
else
    cohorts=()
    for d in "$COHORTS_DIR"/*/; do
        name=$(basename "$d")
        [[ " $EXCLUDE " == *" $name "* ]] && continue
        cohorts+=("$name")
    done
fi
[[ ${#cohorts[@]} -gt 0 ]] || { echo "No cohorts found in $COHORTS_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR"
echo "Model: $MODEL_FP"
echo "Cohorts (${#cohorts[@]}): ${cohorts[*]}"
echo "Train splits: $TRAIN_SPLITS | test split: $TEST_SPLIT | device: $DEVICE x $NUM_GPUS"

extract_features() {  # <cohort> <labels_fp> <split>
    local cohort=$1 labels_fp=$2 split=$3
    local out="$OUT_DIR/$cohort/features_$split"
    if [[ -f "$out/.done" ]]; then
        echo "[$cohort] features_$split already done, skipping"
        return 0
    fi
    rm -rf "$out"  # drop partial output of an interrupted run
    echo "[$cohort] extracting features_$split"
    ethos_linear_prob_features \
        model_fp="$MODEL_FP" \
        input_dir="$TOKENIZED_DIR/$split" \
        labels_fp="$labels_fp" \
        output_dir="$OUT_DIR/$cohort" \
        output_fn="features_$split" \
        average_over_sequence="$AVERAGE_OVER_SEQUENCE" \
        device="$DEVICE" && touch "$out/.done"
}

process_cohort() {  # <cohort>
    local cohort=$1
    local cohort_dir="$COHORTS_DIR/$cohort"
    local labels_fp="$cohort_dir"
    [[ -d "$cohort_dir/data" ]] && labels_fp="$cohort_dir/data"
    if ! compgen -G "$labels_fp/*.parquet" >/dev/null; then
        echo "[$cohort] no parquet files directly in '$labels_fp', skipping" >&2
        return 1
    fi

    mkdir -p "$OUT_DIR/$cohort"
    local split
    for split in $TRAIN_SPLITS $TEST_SPLIT; do
        extract_features "$cohort" "$labels_fp" "$split" || return 1
    done

    if [[ -f "$OUT_DIR/$cohort/logreg/metrics.json" ]]; then
        echo "[$cohort] logistic regression already done, skipping"
        return 0
    fi
    local train_args=()
    for split in $TRAIN_SPLITS; do
        train_args+=("$OUT_DIR/$cohort/features_$split")
    done
    echo "[$cohort] fitting the linear probe"
    python "$SCRIPT_DIR/train_logreg.py" \
        --train_features "${train_args[@]}" \
        --test_features "$OUT_DIR/$cohort/features_$TEST_SPLIT" \
        --output_dir "$OUT_DIR/$cohort/logreg"
}

worker() {  # <gpu index>
    local gpu=$1 i
    for ((i = gpu; i < ${#cohorts[@]}; i += NUM_GPUS)); do
        local cohort=${cohorts[$i]}
        mkdir -p "$OUT_DIR/$cohort"
        if ! CUDA_VISIBLE_DEVICES=$gpu process_cohort "$cohort" >>"$OUT_DIR/$cohort/run.log" 2>&1; then
            echo "[$cohort] FAILED, see $OUT_DIR/$cohort/run.log" | tee -a "$OUT_DIR/failed.txt"
        else
            echo "[$cohort] done"
        fi
    done
}

rm -f "$OUT_DIR/failed.txt"
for ((g = 0; g < NUM_GPUS; g++)); do
    worker "$g" &
done
wait

python - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path

import pandas as pd

rows = []
for fp in sorted(Path(sys.argv[1]).glob("*/logreg/metrics.json")):
    rows.append({"cohort": fp.parent.parent.name, **json.loads(fp.read_text())})
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(Path(sys.argv[1]) / "summary.csv", index=False)
    print(df.to_string(index=False))
else:
    print("No finished cohorts.")
EOF
[[ -f "$OUT_DIR/failed.txt" ]] && { echo "Failed cohorts listed in $OUT_DIR/failed.txt"; exit 1; }
exit 0
