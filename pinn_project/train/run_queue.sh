#!/usr/bin/env bash
# Queues var-window training after v5 finishes, then auto-pushes results.
# Usage: bash run_queue.sh
# Runs in background — safe to close terminal if you use: nohup bash run_queue.sh &

set -e
SOFA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="$SOFA_ROOT/pinn_project/train/queue.log"

echo "[queue] Started at $(date)" | tee "$LOG"

# ── Step 1: Wait for v5 (--use-youngs) to finish ──────────────────────────────
echo "[queue] Waiting for v5 E-gen training to finish..." | tee -a "$LOG"
while pgrep -f "train_liver_pinn.py --use-youngs" > /dev/null; do
    sleep 30
done
echo "[queue] v5 finished at $(date)" | tee -a "$LOG"

# ── Step 2: Start var-window E=1500 training ──────────────────────────────────
echo "[queue] Starting var-window E=1500 training..." | tee -a "$LOG"
cd "$SOFA_ROOT/pinn_project/train"
python3 train_liver_pinn.py --var-window 1.0 --max-rows 80000 2>&1 | tee -a "$LOG"
echo "[queue] Var-window training finished at $(date)" | tee -a "$LOG"

# ── Step 3: Push results ───────────────────────────────────────────────────────
echo "[queue] Pushing results..." | tee -a "$LOG"
cd "$SOFA_ROOT"
git add pinn_project/train/liver_results.txt pinn_project/train/queue.log 2>/dev/null || true
git add pinn_project/train/liver_Egen_v5_loss.png 2>/dev/null || true
git add pinn_project/train/liver_E1500_vw1p0s_v1_loss.png 2>/dev/null || true
git diff --cached --quiet && echo "[queue] Nothing new to commit" && exit 0
git commit -m "$(cat <<'EOF'
Add v5 E-gen and var-window E=1500 training results

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push myfork v23.12
echo "[queue] Done at $(date)" | tee -a "$LOG"
