import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import numpy as np
from ot.lp import wasserstein_1d as wd
from tqdm.auto import tqdm
import json
import time
from emcee import EnsembleSampler
import emcee

plt.style.use("ggplot")

output_root = "mcmc_results_1"
os.makedirs(output_root, exist_ok=True)

a = 2
b = 0.1
sigma_x = 1
cond_no = -1.0

def log_dens(u):
    y = cond_no
    scale = 1 / (2 * sigma_x**2)
    sum_part = (a * (y + b * (u**2 + a**2))) ** 2 + (u**2) / (a**2)
    return -scale * sum_part

no_trials = 10
nsamples = 100000
nwalkers = 16
nsteps = nsamples // nwalkers
ndim = 1
mcmc_samps = np.zeros((no_trials, nwalkers, nsteps, ndim))
sample_no_list = np.load("sample_no_list.npy").tolist()
start = time.perf_counter()
for i in range(no_trials):
    # rng2 = np.random.RandomState(45 + i)
    initial = np.random.randn(nwalkers, ndim)
    # sampler = EnsembleSampler(nwalkers, ndim, log_dens, moves=[
    #     (emcee.moves.DEMove(), 0.8),
    #     (emcee.moves.DESnookerMove(), 0.1),
    #     (emcee.moves.KDEMove(), 0.1),
    # ])
    sampler = EnsembleSampler(nwalkers, ndim, log_dens)
    sampler.run_mcmc(initial, nsteps, progress=True)

    mcmc_samps[i, :, :, :] = sampler.chain
    # print(f"Autocorrelation time: {sampler.get_autocorr_time()[0]} steps")

elapsed = time.perf_counter() - start
seed = 1
choose_samples = 100000
rng = np.random.RandomState(seed)
samps = np.load("rej_samples_1.npy")
# print(samps.reshape(-1).shape)
us_base = rng.randn(choose_samples, 1) * 2
base_wd = wd(
    us_base.reshape(-1),
    samps.reshape(-1),
    p=2,
)

wd_array = np.zeros((no_trials, len(sample_no_list)))

for j in tqdm(range(no_trials)):
    for i, sample_no in enumerate(sample_no_list):
        subsamps = mcmc_samps[j, :, :, :]
        # print(subsamps.shape)
        if sample_no < nwalkers:
            subsamps = subsamps[:sample_no, 0, :]
            # print(subsamps.shape)
            wd1 = wd(
                subsamps.reshape(-1),
                samps.reshape(-1),
                p=2
            )
        else:
            subsamps = subsamps[:, : sample_no // nwalkers, :].reshape(sample_no, ndim)
            wd1 = wd(
                subsamps.reshape(-1),
                samps.reshape(-1),
                p=2
            )

        wd_array[j, i] = wd1

avg_wd_array = np.mean(wd_array / base_wd, axis=0)
std_wd_array = np.std(wd_array / base_wd, axis=0)

np.save(os.path.join(output_root, "avg_wd_mcmc.npy"), avg_wd_array)
np.save(os.path.join(output_root, "std_wd_mcmc.npy"), std_wd_array)
np.save(os.path.join(output_root, "mcmc_samps.npy"), mcmc_samps)
timings = {
    "mcmc_time": elapsed,
    "timestamp": time.time()
}

with open(os.path.join(output_root, "timings.json"), "w") as f:
    json.dump(timings, f, indent=2)