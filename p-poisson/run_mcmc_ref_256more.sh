#!/usr/bin/env bash
# Run 256 additional hMALA chains (chain_625 – chain_880) in mcmc_256.
# Uses 16 batches of 16 parallel ranks each (16 x 16 = 256).
#
# Usage:
#   bash run_mcmc_ref_256more.sh [output_root]
#
# Default:
#   output_root = mcmc_256

OUTPUT_ROOT="${1:-mcmc_256}"
N_PARALLEL=16
N_BATCHES=16
START_OFFSET=625

echo "Output root : $OUTPUT_ROOT"
echo "Chains/batch: $N_PARALLEL"
echo "Batches     : $N_BATCHES"
echo "Total chains: $((N_PARALLEL * N_BATCHES))"
echo "Chain range : $START_OFFSET – $(( START_OFFSET + N_PARALLEL * N_BATCHES - 1 ))"
echo ""

for batch in $(seq 0 $((N_BATCHES - 1))); do
    START_IDX=$(( START_OFFSET + batch * N_PARALLEL ))
    END_IDX=$(( START_IDX + N_PARALLEL - 1 ))
    echo "=== Batch $batch / $((N_BATCHES - 1))  (chains $START_IDX – $END_IDX) ==="
    mpirun -n $N_PARALLEL python3 -u budget_mcmc_runs.py \
        --output_root "$OUTPUT_ROOT" \
        --start_idx   "$START_IDX"
    echo ""
done

echo "All batches done."
