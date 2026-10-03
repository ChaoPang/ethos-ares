#!/usr/bin/env bash
# Linear probing of a pretrained ETHOS checkpoint on a set of ACES/MEDS label cohorts.
#
# Each cohort is a directory of label parquet files (subject_id, prediction_time, boolean_value),
# in a `data/` subfolder or anywhere under it (e.g. train/ and test/): all files found are
# read together, since the split here is by patient, not by the cohort's own train/test folders.
# The same labels are matched against each tokenized MEDS split, and a split only keeps the
# labels of patients it contains. So the pipeline's `train` and `tuning` splits give the
# training features and `held_out` gives the test features.
#
# Prerequisites:
#   - all three MEDS splits tokenized with the SAME vocab (the pipeline script only does train
#     and tuning; held_out needs `ethos_tokenize ... input_dir=<meds>/held_out vocab=<train>
#     out_fn=held_out`, with the same dataset config as train)
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
# Restarting: finished steps are skipped and an interrupted feature extraction continues from
# its last finished chunk, so just run it again. The script ignores SIGHUP, so a dropped SSH
# session does not stop it.
#
# Safety checks:
#   - every tokenized split must be complete (set MEDS_DIR=<dir with train/ tuning/ held_out/ of
#     the source parquet files> to compare shard counts) and use the same vocab as train
#   - each cohort's output folder records the checkpoint, label files, tokenized data and
#     settings it was made with. Rerunning with anything changed stops with an error instead of
#     reusing stale features or a stale fitted model: delete the folder or use a new OUT_DIR.
#   - label prediction_time is compared with the event times as-is. Do not convert it to UTC: in
#     the OMOP MEDS data the `visit` rows are shifted by the UTC offset but the clinical tables
#     are on the labels' clock (see check_label_alignment.py).
#
# Optional: TRAIN_SPLITS="train tuning", TEST_SPLIT=held_out, EXCLUDE="logs", MEDS_DIR,
# GPU_IDS="0 1" (default: all GPUs; one worker per listed GPU, cohorts are spread over them),
# NUM_GPUS, AVERAGE_OVER_SEQUENCE=false (true: mean-pool hidden states instead of using the
# last token).

set -u -o pipefail
trap '' HUP

: "${MODEL_FP:?MODEL_FP must point to the checkpoint, e.g. .../models/<name>/best_model.pt}"
: "${COHORTS_DIR:?COHORTS_DIR must be set}"
: "${TOKENIZED_DIR:?TOKENIZED_DIR must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"

TRAIN_SPLITS=${TRAIN_SPLITS:-"train tuning"}
TEST_SPLIT=${TEST_SPLIT:-held_out}
EXCLUDE=${EXCLUDE:-logs}
MEDS_DIR=${MEDS_DIR:-}
AVERAGE_OVER_SEQUENCE=${AVERAGE_OVER_SEQUENCE:-false}

DEVICE=cuda
if [[ -z "${GPU_IDS:-}" ]]; then
    n_gpus=${NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}
    GPU_IDS=""
    [[ "$n_gpus" -gt 0 ]] && GPU_IDS=$(seq -s ' ' 0 $((n_gpus - 1)))
fi
gpu_ids=($GPU_IDS)
NUM_GPUS=${#gpu_ids[@]}
if [[ "$NUM_GPUS" -eq 0 ]]; then
    DEVICE=cpu
    NUM_GPUS=1
    gpu_ids=("")
fi

COHORTS_DIR=${COHORTS_DIR%/}
TOKENIZED_DIR=${TOKENIZED_DIR%/}
OUT_DIR=${OUT_DIR%/}
MEDS_DIR=${MEDS_DIR%/}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REF_SPLIT=${TRAIN_SPLITS%% *}

[[ -f "$MODEL_FP" ]] || { echo "Checkpoint not found: $MODEL_FP" >&2; exit 1; }

first_file() { find -L "$1" -maxdepth 1 -name "$2" 2>/dev/null | sort | head -n 1; }

validate_split() {  # <split>
    local split=$1 dir="$TOKENIZED_DIR/$1" n_shards
    n_shards=$(find -L "$dir" -maxdepth 1 -name '[0-9]*.safetensors' 2>/dev/null | wc -l | tr -d " ")
    if [[ "$n_shards" -eq 0 ]]; then
        echo "No tokenized shards in '$dir'. Tokenize that split first, reusing the train vocab" \
             "(vocab=$TOKENIZED_DIR/$REF_SPLIT)." >&2
        return 1
    fi
    if [[ ! -f "$dir/static_data.pickle" ]]; then
        echo "'$dir' has no static_data.pickle: its tokenization did not finish." >&2
        return 1
    fi
    if [[ -n "$MEDS_DIR" ]]; then
        local n_src
        n_src=$(find -L "$MEDS_DIR/$split" -maxdepth 1 -name '*.parquet' 2>/dev/null | wc -l | tr -d " ")
        if [[ "$n_src" -ne "$n_shards" ]]; then
            echo "Split '$split': $n_shards tokenized shards but $n_src source shards in" \
                 "$MEDS_DIR/$split. Rerun its tokenization (finished shards are reused)." >&2
            return 1
        fi
    fi
    if [[ "$split" != "$REF_SPLIT" ]]; then
        local v ref
        v=$(first_file "$dir" 'vocab_t*.csv')
        ref=$(first_file "$TOKENIZED_DIR/$REF_SPLIT" 'vocab_t*.csv')
        if [[ -z "$v" || -z "$ref" ]] || ! cmp -s "$v" "$ref"; then
            echo "Split '$split' was not tokenized with the '$REF_SPLIT' vocab (vocab files" \
                 "differ). Retokenize it with vocab=$TOKENIZED_DIR/$REF_SPLIT." >&2
            return 1
        fi
    fi
}

for split in $TRAIN_SPLITS $TEST_SPLIT; do
    validate_split "$split" || exit 1
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
echo "Train splits: $TRAIN_SPLITS | test split: $TEST_SPLIT | device: $DEVICE, workers: $NUM_GPUS"

signature() {  # <labels_fp>: fingerprint of everything the outputs depend on
    python - "$MODEL_FP" "$1" "$TOKENIZED_DIR" "$TRAIN_SPLITS|$TEST_SPLIT|$AVERAGE_OVER_SEQUENCE" \
        "$TRAIN_SPLITS $TEST_SPLIT" <<'EOF'
import hashlib
import os
import sys
from pathlib import Path

model, labels, tok, settings, splits = sys.argv[1:6]
h = hashlib.sha1(settings.encode())


def add(path):
    st = os.stat(path)
    h.update(f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}\n".encode())


add(model)
for root, _, files in os.walk(labels, followlinks=True):
    for name in sorted(files):
        if name.endswith(".parquet"):
            add(os.path.join(root, name))
for split in splits.split():
    d = Path(tok) / split
    shards = sorted(d.glob("[0-9]*.safetensors"))
    h.update(f"{split}|{len(shards)}|{sum(s.stat().st_size for s in shards)}\n".encode())
    for extra in ("static_data.pickle", *sorted(p.name for p in d.glob("vocab_t*.csv"))):
        add(d / extra)
print(h.hexdigest())
EOF
}

check_signature() {  # <cohort> <labels_fp>
    local sig_fp="$OUT_DIR/$1/signature" sig
    sig=$(signature "$2") || return 1
    if [[ -f "$sig_fp" && "$(cat "$sig_fp")" != "$sig" ]]; then
        echo "[$1] the outputs in $OUT_DIR/$1 were made with a different checkpoint, labels," \
             "tokenized data or settings. Delete that folder or use another OUT_DIR." >&2
        return 1
    fi
    echo "$sig" >"$sig_fp"
}

extract_features() {  # <cohort> <labels_fp> <split>
    local cohort=$1 labels_fp=$2 split=$3
    local out="$OUT_DIR/$cohort/features_$split"
    if [[ -f "$out/.done" ]]; then
        echo "[$cohort] features_$split already done, skipping"
        return 0
    fi
    echo "[$cohort] extracting features_$split (continues from any finished chunks)"
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
    # labels may be spread over subfolders (data/, or train/ + test/); all of them are read
    local n_label_files
    n_label_files=$(find -L "$labels_fp" -name '*.parquet' 2>/dev/null | wc -l | tr -d " ")
    if [[ "$n_label_files" -eq 0 ]]; then
        echo "[$cohort] no parquet files under '$labels_fp', skipping" >&2
        return 1
    fi
    echo "[$cohort] labels: $n_label_files parquet file(s) under $labels_fp"

    mkdir -p "$OUT_DIR/$cohort"
    check_signature "$cohort" "$labels_fp" || return 1
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

worker() {  # <worker index>
    local w=$1 i
    for ((i = w; i < ${#cohorts[@]}; i += NUM_GPUS)); do
        local cohort=${cohorts[$i]}
        mkdir -p "$OUT_DIR/$cohort"
        if ! CUDA_VISIBLE_DEVICES=${gpu_ids[$w]} process_cohort "$cohort" \
            >>"$OUT_DIR/$cohort/run.log" 2>&1; then
            echo "$cohort" >>"$OUT_DIR/failed.txt"
            echo "[$cohort] FAILED"
        else
            echo "[$cohort] done"
        fi
    done
}

rm -f "$OUT_DIR/failed.txt"
for ((w = 0; w < NUM_GPUS; w++)); do
    worker "$w" &
done
wait

python - "$OUT_DIR" <<'EOF'
import json
import sys
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

if [[ -f "$OUT_DIR/failed.txt" ]]; then
    echo
    echo "Failed cohorts (last lines of each run.log):"
    while read -r cohort; do
        echo "--- $cohort ($OUT_DIR/$cohort/run.log)"
        tr '\r' '\n' <"$OUT_DIR/$cohort/run.log" | grep -vE 'it/s\]|Computing features' | tail -n 3
    done <"$OUT_DIR/failed.txt"
    exit 1
fi
exit 0
