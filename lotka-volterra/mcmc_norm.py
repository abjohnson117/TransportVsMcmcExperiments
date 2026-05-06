import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import time
import json
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.optimize import minimize
from ot.sliced import sliced_wasserstein_distance as swd
from triangular_transport.mcmc.adaptive_mcmc import AdaptiveMCMC
from triangular_transport.flows.dataloaders import log_normal_reference_sampler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "true_samps"))
from lv import LV

jax.config.update("jax_enable_x64", True)

output_root = "mcmc_norm_results"
os.makedirs(output_root, exist_ok=True)

# --- Setup posterior ---
no_samples = 1
u_true = jnp.array([0.83194674, 0.04134147, 1.0823151, 0.03991483])
sigma = 1.0
mean = np.load("log_y_mean.npy")
std = np.load("log_y_std.npy")

lv_sampler = LV(
    seed=0,
    no_samples=no_samples,
    prior_sampler=log_normal_reference_sampler,
    likelihood_sampler=log_normal_reference_sampler,
    normalize=False,
    u_true=u_true,
    sigma=sigma,
    dt0=0.1,
    log_y_mean=mean,
    log_y_std=std,
)

yobs = np.load("true_samps/y_obs.npy")

pi = lv_sampler.posterior
log_pi = lv_sampler.log_posterior

def post(x):
    return pi(yobs, x)

@jax.jit
def log_post(x):
    return log_pi(yobs, x)

def neg_post(x):
    return -log_pi(yobs, x)

def alpha(x, w, log_density):
    log_w = log_density(w)
    log_x = log_density(x)
    return jnp.minimum(0.0, log_w - log_x)

# --- MAP estimate with function evaluation count ---
n_restarts = 25
u_dim = 4
init_key = random.key(seed=11)
x_inits = []
for i in range(n_restarts):
    init_key, subkey = random.split(init_key)
    x_inits.append(
        random.normal(subkey, shape=(u_dim,)) * lv_sampler.std_prior + lv_sampler.mu_base
    )

best_val = jnp.inf
best_result = None
total_map_nfev = 0
for i, x_init in enumerate(tqdm(x_inits, desc="MAP restarts")):
    result = minimize(neg_post, x_init, method="BFGS", tol=1e-8)
    val = neg_post(result.x)
    print(f"Init {i}: neg_post = {val:.4f}, exp(x) = {jnp.exp(result.x)}, nfev = {int(result.nfev)}")
    if val < best_val:
        best_val = val
        best_result = result

x0 = best_result.x
total_map_nfev = best_result.nfev + best_result.njev
print(f"\nBest x0: {jnp.exp(x0)}  (neg_post = {best_val:.4f})")
print(f"Total MAP function evaluations across all restarts: {total_map_nfev}")

# --- MCMC ---
no_trials = 10
nsteps = 100000
burn_in = 1
ndim = 4
mcmc_samps = np.zeros((no_trials, nsteps, ndim))

start = time.perf_counter()
for i in range(no_trials):
    print(f"Running trial {i+1}/{no_trials}...")
    iter_key = random.key(i+10)
    x0_pert = x0 +  0.008 * random.normal(key=iter_key, shape=x0.shape)
    adapt_mcmc = AdaptiveMCMC(
        target_density=post,
        log_density=log_post,
        alpha_function=alpha,
        seed=np.random.choice(1000000),
        train_dim=ndim,
        steps=nsteps,
        name="Adaptive - Conditional",
        std_err=0.05,
        iter_step_adapt=100,
        cond_no=yobs,
        burn_in=burn_in,
        x0=x0_pert,
    )
    adapt_mcmc.fit(print_every=100, log_update=False)
    mcmc_samps[i, :, :] = adapt_mcmc.samples

elapsed = time.perf_counter() - start
print(f"MCMC elapsed time: {elapsed:.2f}s")

# --- SWD vs true samples (as in sample_convergence.py) ---
swd_seed = 42
n_projections = 2048
gen_seed = 1
gen_key = random.key(gen_seed)

true_samps = np.exp(np.load("true_samps/true_us_obs.npy")[::20, :])

us_base = log_normal_reference_sampler(
    key=gen_key,
    shape=true_samps.shape,
    mu=jnp.array([-0.125, -3.0, -0.125, -3.0]),
    sigma=1 / jnp.sqrt(2),
)

base_swd = swd(
    np.asarray(us_base),
    np.asarray(true_samps),
    a=np.ones(len(us_base)) / len(us_base),
    b=np.ones(len(true_samps)) / len(true_samps),
    seed=swd_seed,
    n_projections=n_projections,
    p=2,
)
print(f"Base SWD (prior vs true): {base_swd}")

sample_no_list = [2**i for i in range(1, 15)]
sample_no_list += [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

wd_array = np.zeros((no_trials, len(sample_no_list)))
for j in tqdm(range(no_trials), desc="SWD trials"):
    for i, sample_no in enumerate(sample_no_list):
        us_gen = np.exp(mcmc_samps[j, :sample_no, :])
        wd_array[j, i] = swd(
            np.asarray(us_gen),
            np.asarray(true_samps),
            a=np.ones(len(us_gen)) / len(us_gen),
            b=np.ones(len(true_samps)) / len(true_samps),
            seed=swd_seed,
            n_projections=n_projections,
            p=2,
        ) / base_swd

avg_wd_array = np.mean(wd_array, axis=0)
std_wd_array = np.std(wd_array, axis=0)

np.save(os.path.join(output_root, "avg_wd_mcmc.npy"), avg_wd_array)
np.save(os.path.join(output_root, "std_wd_mcmc.npy"), std_wd_array)
np.save(os.path.join(output_root, "mcmc_samps.npy"), mcmc_samps)
np.save(os.path.join(output_root, "sample_no_list.npy"), np.array(sample_no_list))

timings = {
    "mcmc_time": elapsed,
    "total_map_nfev": total_map_nfev,
    "timestamp": time.time(),
}
with open(os.path.join(output_root, "timings.json"), "w") as f:
    json.dump(timings, f, indent=2)

print(f"\nTotal MAP function evaluations: {total_map_nfev}")
print(f"Results saved to {output_root}/")
