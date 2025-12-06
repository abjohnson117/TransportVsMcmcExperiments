ps aux
BLOCK_SIZE=25        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=125      # 0 for chains 0–49, then 50, 100, ...
YS_PATH="training_dataset/solutions_grid_delta.npy"
OUTPUT_ROOT="mcmc_chains"
nohup bash -c "
TOTAL=${BLOCK_SIZE}
BATCH_SIZE=${BATCH_SIZE}
GLOBAL_OFFSET=${GLOBAL_OFFSET}
YS_PATH=\"${YS_PATH}\"
OUTPUT_ROOT=\"${OUTPUT_ROOT}\"

for ((batch=0; batch< TOTAL / BATCH_SIZE; batch++)); do
    local_start=\$((batch * BATCH_SIZE))
    start_idx=\$((GLOBAL_OFFSET + local_start))

    echo \"Starting batch \$batch, global start_idx=\$start_idx\"

    mpirun -np \${BATCH_SIZE} python3 poisson.py \
        --data_path \"\${YS_PATH}\" \
        --output_root \"\${OUTPUT_ROOT}\" \
        --start_idx \"\${start_idx}\"

    echo \"Finished batch \$batch (chains \$start_idx .. \$((start_idx + BATCH_SIZE - 1)))\"
done
" > hmala_block_${GLOBAL_OFFSET}.log 2>&1 &
