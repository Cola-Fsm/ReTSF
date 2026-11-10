#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd -- "$PROJECT_ROOT"

DATASET="ETTm2"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/CAF_full/$RUN_TAG/$DATASET}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/CAF/full/$RUN_TAG/$DATASET}"
unset PYTORCH_CUDA_ALLOC_CONF
mkdir -p -- "$OUTPUT_ROOT" "$LOG_ROOT"

# horizon warmup_epochs gain_epochs learning_rate patch_len hidden rank_weight rank_margin rank_min_gap
while read -r horizon warmup gain_epochs learning_rate patch_len hidden rank_weight rank_margin rank_min_gap; do
    run_tag="full_${DATASET}_H${horizon}_seed2025_${RUN_TAG}"
    printf '\n[CAF full] Dataset=%s | Horizon=%s | Seed=2025\n' "$DATASET" "$horizon"

    "$PYTHON" -u "$PROJECT_ROOT/run_caf.py" \
        --datasets "$DATASET" \
        --horizons "$horizon" \
        --dataset_root "$DATASET_ROOT" \
        --seq_len 320 \
        --batch_size 128 \
        --train_epochs 20 \
        --patience 3 \
        --target_metric MAE \
        --seed 2025 \
        --device "$DEVICE" \
        --caf_gain_lr 0.001 \
        --caf_budget -1 \
        --caf_train_fraction 0.8 \
        --caf_candidates 3 \
        --num_workers 0 \
        --threads 4 \
        --log_every_batches 50 \
        --caf_warmup_epochs "$warmup" \
        --caf_gain_epochs "$gain_epochs" \
        --learning_rate "$learning_rate" \
        --caf_patch_len "$patch_len" \
        --caf_hidden "$hidden" \
        --caf_rank_weight "$rank_weight" \
        --caf_rank_margin "$rank_margin" \
        --caf_rank_min_gap "$rank_min_gap" \
        --run_tag "$run_tag" \
        --output_dir "$OUTPUT_ROOT" \
        2>&1 | tee "$LOG_ROOT/H${horizon}.log"
done <<'BEST_PARAMS'
96 3 12 0.0005 24 32 0 0.005 0.0025
192 2 4 0.0005 48 64 0.05 0.005 0.005
336 2 8 0.0005 48 128 0.1 0.02 0.01
720 3 12 0.0005 24 32 0.2 0.005 0.01
BEST_PARAMS

printf '\nCompleted %s: horizons 96, 192, 336, 720.\nResults: %s\nLogs: %s\n' \
    "$DATASET" "$OUTPUT_ROOT" "$LOG_ROOT"
