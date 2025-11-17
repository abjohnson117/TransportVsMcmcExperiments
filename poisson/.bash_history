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
