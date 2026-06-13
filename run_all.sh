#!/usr/bin/env bash
# run_all.sh — run the full pipeline in order.
#
# Step 1: raw .sif frames -> count rates -> diffraction efficiencies (HDF5)
# Step 2: MCMC fit of the grating profile + diagnostic figures
#
# Usage:
#   bash run_all.sh
#
# This assumes `uv` is installed and you are in the repository root (so that the
# relative paths in config.py resolve).

set -euo pipefail

echo "=========================================="
echo " STEP 1/2: computing diffraction efficiencies"
echo "=========================================="
uv run step1_compute_efficiencies.py

echo
echo "=========================================="
echo " STEP 2/2: MCMC grating-profile fit"
echo "=========================================="
uv run step2_fit_gratings.py

echo
echo "All done. Figures are in the plots/ directory."
