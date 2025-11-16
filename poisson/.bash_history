nohup mpirun -np 25 python3 poisson.py --data_path data_50.npy --output_root mcmc_median/ > poisson_med.log 2>&1 &
exit
