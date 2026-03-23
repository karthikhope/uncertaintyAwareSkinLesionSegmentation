#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Uncertainty-Aware Skin Lesion Segmentation — Full Pipeline"
echo "============================================================"
echo ""
echo "Available models: unet, attention_unet, resunet"
echo ""
read -p "Enter model name: " MODEL

if [[ "$MODEL" != "unet" && "$MODEL" != "attention_unet" && "$MODEL" != "resunet" ]]; then
    echo "ERROR: Invalid model '$MODEL'. Must be one of: unet, attention_unet, resunet"
    exit 1
fi

CHECKPOINT="best_model_${MODEL}.pth"
HISTORY="training_history_${MODEL}.csv"
REPORTS_DIR="reports/${MODEL}"
REPORTS_DET_DIR="reports/${MODEL}_deterministic"

echo ""
echo "Model:       $MODEL"
echo "Checkpoint:  src/$CHECKPOINT"
echo "Reports:     $REPORTS_DIR/"
echo ""

STEP=0
total_steps=9

step() {
    STEP=$((STEP + 1))
    echo ""
    echo "============================================================"
    echo "  [$STEP/$total_steps] $1"
    echo "============================================================"
}

# ── Step 1: Training ─────────────────────────────────────────────
step "Training $MODEL (100 epochs)"
cd "$PROJECT_ROOT/src"
python train.py --model "$MODEL" --use_isic --epochs 100
cd "$PROJECT_ROOT"

# ── Step 2: Bayesian evaluation (MC Dropout, T=20) ──────────────
step "Bayesian evaluation (mc_passes=20) → $REPORTS_DIR/"
cd "$PROJECT_ROOT/src"
python eval.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --use_isic \
    --mc_passes 20 \
    --model_name "$MODEL" \
    --reports_dir "../$REPORTS_DIR"
cd "$PROJECT_ROOT"

# ── Step 3: Deterministic evaluation (mc_passes=0) ──────────────
step "Deterministic evaluation (mc_passes=0) → $REPORTS_DET_DIR/"
cd "$PROJECT_ROOT/src"
python eval.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --use_isic \
    --mc_passes 0 \
    --model_name "${MODEL}_deterministic" \
    --reports_dir "../$REPORTS_DET_DIR"
cd "$PROJECT_ROOT"

# ── Step 4: Severity degradation plots ───────────────────────────
step "Severity plots (Dice/IoU/ECE/pECE vs severity)"
python scripts/plot_results.py \
    --results_csv "$REPORTS_DIR/eval_results.csv" \
    --reports_dir "$REPORTS_DIR"

# ── Step 5: Statistical significance tests (~29 min) ─────────────
step "Statistical tests (Wilcoxon + bootstrap ECE)"
python scripts/stat_tests.py \
    --model "$MODEL" \
    --checkpoint "src/$CHECKPOINT" \
    --reports_dir "$REPORTS_DIR"

# ── Step 6: Training curves ──────────────────────────────────────
step "Training curves"
python scripts/plot_training_curves.py \
    --csv "src/$HISTORY" \
    --output "$REPORTS_DIR/training_curves.png"

# ── Step 7: Uncertainty map grids ────────────────────────────────
step "Uncertainty map grids (clean vs corrupted)"
for CORRUPTION in gaussian_noise gaussian_blur motion_blur contrast_shift; do
    echo "  → $CORRUPTION severity=5"
    python scripts/visualize_uncertainty_maps.py \
        --model "$MODEL" \
        --checkpoint "src/$CHECKPOINT" \
        --corruption "$CORRUPTION" \
        --severity 5 \
        --reports_dir "$REPORTS_DIR"
done

# ── Step 8: Failure case gallery ─────────────────────────────────
step "Failure case galleries"
echo "  → clean"
python scripts/failure_gallery.py \
    --model "$MODEL" \
    --checkpoint "src/$CHECKPOINT" \
    --reports_dir "$REPORTS_DIR"

echo "  → gaussian_noise severity=5"
python scripts/failure_gallery.py \
    --model "$MODEL" \
    --checkpoint "src/$CHECKPOINT" \
    --corruption gaussian_noise \
    --severity 5 \
    --reports_dir "$REPORTS_DIR"

# ── Step 9: Summary ──────────────────────────────────────────────
step "Done!"
echo ""
echo "  Model:        $MODEL"
echo "  Checkpoint:   src/$CHECKPOINT"
echo "  History CSV:  src/$HISTORY"
echo "  Reports:      $REPORTS_DIR/"
echo "  Deterministic:$REPORTS_DET_DIR/"
echo ""
echo "  Generated files:"
ls -1 "$REPORTS_DIR/" | sed 's/^/    /'
echo ""
echo "============================================================"
echo "  All steps complete for $MODEL"
echo "============================================================"
