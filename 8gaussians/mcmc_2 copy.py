import os
import matplotlib.pyplot as plt
import numpy as np
from ot.lp import wasserstein_1d as wd
from tqdm.auto import tqdm
import json
import time
from emcee import EnsembleSampler

plt.style.use("ggplot")

output_root = "mcmc_results_2"
os.makedirs(output_root, exist_ok=True)

cond_no = 2.45

def eight_gaussians_cond(u):
    scale = 4.0
    u = np.asarray(u)
    if u.ndim > 0:
        u = u.item()
    y = cond_no
    y = float(np.asarray(y).item())
    centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
    centers = np.array([((scale * x) / np.sqrt(2), (scale * y) / (np.sqrt(2))) for x, y in centers], dtype="float32")
    factor = 1 / np.sqrt(np.pi)
    x = np.array([y, u])
    x = np.full((8, 2), fill_value=x)
    norm_part = np.sum((x - centers) * (x - centers), axis=1)
    exp_part = np.exp(-scale * norm_part)
    sum_part = np.sum(exp_part)
    return (factor * sum_part).item()

def log_eight_gaussians_cond(u):
    return float(np.log(eight_gaussians_cond(u)))

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
    sampler = EnsembleSampler(nwalkers, ndim, log_eight_gaussians_cond)
    sampler.run_mcmc(initial, nsteps, progress=True)

    mcmc_samps[i, :, :, :] = sampler.chain
    # print(f"Autocorrelation time: {sampler.get_autocorr_time()[0]} steps")

elapsed = time.perf_counter() - start
seed = 1
choose_samples = 100000
rng = np.random.RandomState(seed)
samps = np.load("rej_samples_2.npy")
# print(samps.reshape(-1).shape)
us_base = rng.randn(choose_samples, 1)
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