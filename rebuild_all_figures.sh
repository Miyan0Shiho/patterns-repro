#!/usr/bin/env bash
# Rebuild all 5 figures from the included summary data.
# After this script finishes, every evidence/<exp>/figures/ should contain
# a fresh 4-format rendering of the corresponding paper figure.
#
# Usage:  bash rebuild_all_figures.sh
# Or:     ./rebuild_all_figures.sh   (after chmod +x)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "Rebuilding all figures"
echo "============================================================"
echo

run() {
    local label="$1"
    local script="$2"
    echo "--- $label ---"
    if [ ! -f "$script" ]; then
        echo "  SKIP: $script not found"
        return 0
    fi
    if python3 "$script"; then
        echo "  OK: $label"
    else
        echo "  FAIL: $label (exit $?)"
        return 1
    fi
    echo
}

# Order matches the manuscript figure numbering.
run "Fig 1  (Conceptual framework)"    "evidence/00-fig1-conceptual/code/build_concept_image.py"
run "Fig 2  (Task heterogeneity)"      "evidence/01-macro-heterogeneity/code/Fig2_TaskHeterogeneity.py"
run "Fig 3  (Model scale)"             "evidence/01-macro-heterogeneity/code/Fig3_ScalingLaws.py"
run "Fig 4  (Reflection scatter)"      "evidence/02-state-divergence/code/Fig4_ReflectionScatter.py"
run "Fig 5  (Logit Lens, 4 conditions)" "evidence/03-logit-lens/code/Fig5_LogitLens.py"

echo "============================================================"
echo "Done. Outputs are in evidence/<exp>/figures/ (4 formats each)."
echo "============================================================"
