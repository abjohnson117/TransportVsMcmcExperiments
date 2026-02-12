import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import numpy as np
from ot.lp import wasserstein_1d as wd
from tqdm.auto import tqdm
import json
import time
from jax import jit, random
import jax.numpy as jnp
from triangular_transport.mcmc.adaptive_mcmc import AdaptiveMCMC

plt.style.use("ggplot")

output_root = "mcmc_results_0"
os.makedirs(output_root, exist_ok=True)

cond_no = 0.0
scale = 0.3

@jit
def tanh_density(u):
    shift = jnp.tanh(cond_no)
    base = (1.0 / scale) * jnp.exp(-(u - shift) / scale)
    return jnp.where(u < shift, 0.0, base)

def log_density(u):
    return np.log(tanh_density(cond_no, u))

def alpha(x, w, log_density):
    log_w = log_density(w)
    log_x = log_density(x)
    return jnp.minimum(0.0, log_w - log_x)

no_trials = 10
nsamples = 100000
nsteps = nsamples
burn_in = 1
ndim = 1
mcmc_samps = np.zeros((no_trials, nsteps, ndim))
sample_no_list = np.load("sample_no_list.npy").tolist()
start = time.perf_counter()
for i in range(no_trials):
    adapt_mcmc = AdaptiveMCMC(
        target_density=tanh_density,
        alpha_function=alpha,
        seed=np.random.choice(1000000),
        train_dim=1,
        steps=nsteps,
        name="Adaptive - Conditional",
        std_err=0.6,
        iter_step_adapt=1,
        cond_no=cond_no,
        burn_in=burn_in,
    )
    adapt_mcmc.fit(print_every=20000)

    mcmc_samps[i, :, :] = adapt_mcmc.samples
    # print(f"Autocorrelation time: {sampler.get_autocorr_time()[0]} steps")

elapsed = time.perf_counter() - start
seed = 1
choose_samples = 100000
rng = np.random.RandomState(seed)
samps = np.load("samps_0.npy")
us_base = rng.randn(nsamples,)
base_wd = wd(
    us_base.reshape(-1),
    samps.reshape(-1),
    p=2,
)

wd_array = np.zeros((no_trials, len(sample_no_list)))
for j in tqdm(range(no_trials)):
    for i, sample_no in enumerate(sample_no_list):
        subsamps = mcmc_samps[j, :, :]
        subsamps = subsamps[:sample_no, :]
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