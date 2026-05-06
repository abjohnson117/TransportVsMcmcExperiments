#!/usr/bin/env bash
# Run all three poisson_mcmc_* scripts sequentially for start_idx 0..9.
# Output folders: mcmc_main_converge_samps, mcmc_med_converge_samps, mcmc_98_converge_samps
# Each chain lands in <output_root>/chain_<start_idx:03d>/

set -euo pipefail

SCRIPTS=(
    "poisson_mcmc_main.py mcmc_main_converge_samps"
    "poisson_mcmc_med.py  mcmc_med_converge_samps"
    "poisson_mcmc_98.py   mcmc_98_converge_samps"
)

for entry in "${SCRIPTS[@]}"; do
    script=$(echo "$entry" | awk '{print $1}')
    outdir=$(echo "$entry" | awk '{print $2}')
    for start_idx in $(seq 0 9); do
        echo "====== $script  start_idx=$start_idx  output_root=$outdir ======"
        python3 -u "$script" --output_root "$outdir" --start_idx "$start_idx"
    done
done

echo "All runs complete."
