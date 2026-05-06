#!/usr/bin/env bash
# Run 25 batches of 25 parallel hMALA chains sequentially.
# Each batch uses mpirun -n 25 so chains 0-24, 25-49, ..., 600-624 are produced.
# Observations are drawn from the test split of the training dataset (index = chain rank).
# Total: 25 x 25 = 625 chains.
#
# Usage:
#   bash run_mcmc_ref.sh [output_root]
#
# Default:
#   output_root = mcmc_ref

OUTPUT_ROOT="${1:-mcmc_ref}"
N_PARALLEL=25
N_BATCHES=25

echo "Output root : $OUTPUT_ROOT"
echo "Chains/batch: $N_PARALLEL"
echo "Batches     : $N_BATCHES"
echo "Total chains: $((N_PARALLEL * N_BATCHES))"
echo ""

for batch in $(seq 0 $((N_BATCHES - 1))); do
    START_IDX=$((batch * N_PARALLEL))
    echo "=== Batch $batch / $((N_BATCHES - 1))  (chains $START_IDX – $((START_IDX + N_PARALLEL - 1))) ==="
    mpirun -n $N_PARALLEL python3 -u budget_mcmc_runs.py \
        --output_root "$OUTPUT_ROOT" \
        --start_idx   "$START_IDX"
    echo ""
done

echo "All batches done."
