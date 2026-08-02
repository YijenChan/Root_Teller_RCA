#!/bin/bash
# Aggregate evaluation across all three benchmarks.
# Prints per-subset R@k / MRR alongside the paper's reported numbers
# (Qwen-3.6-Plus backbone — the default configuration of this release).
#
# Usage: bash final_eval.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PYTHON=${PYTHON:-python3}

echo "=================================================================="
echo "  RCLAgent Evaluation Report — $(date)"
echo "=================================================================="
echo ""

# ── AIOPS 2022 ────────────────────────────────────────────────────────
echo "### AIOPS 2022 (6 subsets, Qwen-3.6-Plus)"
echo ""
printf "  %-4s | %-22s | %8s | %13s\n" "Lbl" "Subset" "Our MRR" "Paper MRR"
echo "  -----|------------------------|----------|--------------"

declare -A AIOPS_LABEL=(
  ["2022-03-20-cloudbed2"]="A"  ["2022-03-20-cloudbed3"]="B"
  ["2022-03-21-cloudbed1"]="Γ"  ["2022-03-21-cloudbed2"]="Δ"
  ["2022-03-21-cloudbed3"]="E"  ["2022-03-24-cloudbed3"]="Z"
)
declare -A AIOPS_PAPER_MRR=(
  ["2022-03-20-cloudbed2"]="66.67"  ["2022-03-20-cloudbed3"]="81.39"
  ["2022-03-21-cloudbed1"]="67.97"  ["2022-03-21-cloudbed2"]="66.78"
  ["2022-03-21-cloudbed3"]="82.22"  ["2022-03-24-cloudbed3"]="73.67"
)

aiops_subsets=(
  "2022-03-20-cloudbed2" "2022-03-20-cloudbed3"
  "2022-03-21-cloudbed1" "2022-03-21-cloudbed2"
  "2022-03-21-cloudbed3" "2022-03-24-cloudbed3"
)

for subset in "${aiops_subsets[@]}"; do
  label=${AIOPS_LABEL[$subset]}
  if [ ! -d "data/$subset/result" ]; then
    printf "  %-4s | %-22s | %8s | %13s\n" "$label" "$subset" "(missing)" "${AIOPS_PAPER_MRR[$subset]}"
    continue
  fi
  mrr=$($PYTHON -c "
import sys, io, contextlib; sys.path.insert(0, '.')
from evaluate import evaluate
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    evaluate('data/$subset', 'result')
for line in buf.getvalue().split('\n'):
    if 'MRR' in line and ':' in line:
        val = line.split(':')[1].strip().split()[0]
        print(f'{float(val)*100:.2f}')
        break
else:
    print('N/A')
" 2>/dev/null)
  printf "  %-4s | %-22s | %8s | %13s\n" "$label" "$subset" "$mrr" "${AIOPS_PAPER_MRR[$subset]}"
done

echo ""
echo "  AIOPS 2022 overall (weighted avg, paper):  R@1 65.15 | R@3 78.79 | R@5 86.36 | R@10 95.45 | MRR 73.24"
echo ""

# ── Augmented-TrainTicket (Nezha-30) ──────────────────────────────────
echo "### Augmented-TrainTicket (Nezha, 2023-01-30, Qwen-3.6-Plus)"
echo ""
if [ -d "data/nezha-2023-01-30/result" ]; then
  $PYTHON evaluate.py data/nezha-2023-01-30 result 2>/dev/null | grep -E "R@|MRR" | sed 's/^/  /'
else
  echo "  (result directory missing — run coordinator first)"
fi
echo ""
echo "  Paper: R@1 82.35 | R@3 88.24 | R@5 94.12 | R@10 94.12 | MRR 86.47"
echo ""

# ── RCAEval RE2-OB ────────────────────────────────────────────────────
echo "### RCAEval RE2-OB (Qwen-3.6-Plus)"
echo ""
if [ -d "data/re2ob/result" ]; then
  $PYTHON evaluate.py data/re2ob result 2>/dev/null | grep -E "R@|MRR" | sed 's/^/  /'
else
  echo "  (result directory missing — run coordinator first)"
fi
echo ""
echo "  Paper: R@1 56.67 | R@3 80.00 | R@5 86.67 | R@10 100.00 | MRR 71.03"
echo ""
echo "=================================================================="
