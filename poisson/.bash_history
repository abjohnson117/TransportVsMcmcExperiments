nohup mpirun -np 25 python3 poisson.py --data_path data_50.npy --output_root mcmc_median/ > poisson_med.log 2>&1 &
exit
cd /workspace/
rm -rf .cache/dijitso
mkdir -p .cache/dijitso/log
pwd
python3 - << 'EOF'
import hippylib as hp
from hippylib.modeling import ExpressionModule
print("JIT compile ok")
EOF

python3 - << 'EOF'
ipmort hippylib as hp

python3 poisson.py   --data_path data_50.npy   --output_root mcmc_median   > poisson_med.log 2>&1
python3 poisson.py --data_path data_50.npy --output_root mcmc_median/ > poisson_med.log 2>&1
pip install packaging
python3 poisson.py --data_path data_50.npy --output_root mcmc_median/ > poisson_med.log 2>&1
nohup mpirun -np 25 python3 poisson.py --data_path data_50.npy --output_root mcmc_median/ > poisson_med.log 2>&1 &
nohup mpirun -np 25 python3 poisson.py --data_path data_98.npy --output_root mcmc_98/ > poisson_98.log 2>&1 &
nohup mpirun -np 25 python3 poisson.py --data_path data_50.npy --output_root mcmc_median/ > poisson_med.log 2>&1 &
python3 qoi_tracing.py 
pip install tqdm
python3 qoi_tracing.py 
python3 qoi_tracing.py 
python3 qoi_tracing.py 
echo $HOME
mkdir -p ~/.local
nohup bash -c '
TOTAL=500
BATCH_SIZE=25
YS_PATH="training_dataset/solutions_grid_delta.npy"
OUTPUT_ROOT="mcmc_chains"

for ((batch=0; batch< TOTAL / BATCH_SIZE; batch++)); do
    start_idx=$((batch * BATCH_SIZE))
    echo "Starting batch $batch, start_idx=$start_idx"

    mpirun -np ${BATCH_SIZE} python3 poisson.py \
        --data_path "${YS_PATH}" \
        --output_root "${OUTPUT_ROOT}" \
        --start_idx "${start_idx}"

    echo "Finished batch $batch"
done
' > hmala_500.log 2>&1 &
nohup bash -c '
TOTAL=500
BATCH_SIZE=25
YS_PATH="training_dataset/solutions_grid_delta.npy"
OUTPUT_ROOT="mcmc_chains"

for ((batch=0; batch< TOTAL / BATCH_SIZE; batch++)); do
    start_idx=$((batch * BATCH_SIZE))
    echo "Starting batch $batch, start_idx=$start_idx"

    mpirun -np ${BATCH_SIZE} python3 poisson.py \
        --data_path "${YS_PATH}" \
        --output_root "${OUTPUT_ROOT}" \
        --start_idx "${start_idx}"

    echo "Finished batch $batch"
done
' > hmala_500.log 2>&1 &
BLOCK_SIZE=50        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=0      # 0 for chains 0–49, then 50, 100, ...
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
ps -ef | grep nohup
kill -9 1008
ps -ef | grep nohup
kill -9 12
ps -ef | grep nohup
ps aux
BLOCK_SIZE=50        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=50      # 0 for chains 0–49, then 50, 100, ...
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
ps aux
ps aux
ps aux
ps aux
BLOCK_SIZE=50        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=100      # 0 for chains 0–49, then 50, 100, ...
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
BLOCK_SIZE=50        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=100      # 0 for chains 0–49, then 50, 100, ...
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
BLOCK_SIZE=50        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=150      # 0 for chains 0–49, then 50, 100, ...
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
ps aux
ps aux
BLOCK_SIZE=25        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=175      # 0 for chains 0–49, then 50, 100, ...
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
BLOCK_SIZE=100        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=200      # 0 for chains 0–49, then 50, 100, ...
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
ps aux
ps aux
ps aux
BLOCK_SIZE=100        # how many chains in this block
BATCH_SIZE=25        # how many chains in parallel per mpirun
GLOBAL_OFFSET=300      # 0 for chains 0–49, then 50, 100, ...
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
pip install scikit-image
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
python3 sl.py 
