import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
from jax import jit
import jax
from ksd import compute_ksd_jax
from jax import vmap, grad
import matplotlib.pyplot as plt
import numpy as np
from ot.sliced import sliced_wasserstein_distance as swd
from tqdm.auto import tqdm
from scipy.optimize import minimize_scalar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()

def rejection_sample(n, M, tau=5.0, bounds=(-50, 50), batch=4096, seed=0):
    rng = np.random.default_rng(seed)

    log_m = np.log(M)
    samples = []
    n_accepted = 0
    n_total = 0

    while n_accepted < n:
        # propose a batch from q
        u_prop = rng.normal(loc=0.0, scale=tau, size=batch)

        # log acceptance ratios
        log_alpha = np.array([log_dens(u) - log_gauss(u) - log_m for u in u_prop])

        # accept with probability exp(log_alpha) (clipped at 1 for safety)
        alpha = np.exp(np.minimum(log_alpha, 0.0))
        u = rng.uniform(size=batch)
        accept = u < alpha

        accepted = u_prop[accept]
        samples.append(accepted)

        n_total += batch
        n_accepted += accepted.size

    samples = np.concatenate(samples)[:n]
    accept_rate = n / n_total

    return samples, accept_rate

output_root = "rej_results"
output_dir = os.path.join(output_root, f"_{args.run_id}")
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
nsamples = 1500
tau = 5.0
a = 2
b = 0.1
sigma_x = 1
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-5, high=0.4, size=(budget, ))

rej_samps = np.zeros((budget, nsamples))

for i, y in enumerate(tqdm(conditioning_ys)):
    def log_dens(u):
        scale = 1 / (2 * sigma_x**2)
        sum_part = (a * (y + b * (u**2 + a**2))) ** 2 + (u**2) / (a**2)
        return -scale * sum_part

    def log_gauss(u):
        # u = float(u)
        # IMPORTANT: include normalization if you want M for rejection sampling
        return -0.5*(u*u)/(tau*tau) - 0.5*np.log(2*np.pi*tau*tau)

    def neg_log_ratio(u):
        return -(log_dens(u) - log_gauss(u))
    
    res = minimize_scalar(neg_log_ratio, bounds=(-50, 50), method="bounded")
    u_star = res.x
    max_log_ratio = -res.fun
    M = np.exp(max_log_ratio)

    samps, accept_rate = rejection_sample(nsamples, M=M, tau=tau, seed=i)
    rej_samps[i, :] = samps

print("Saving results...")
np.save(os.path.join(output_dir, "conditioning_ys.npy"), conditioning_ys)
np.save(os.path.join(output_dir, "rej_samps.npy"), rej_samps)
print("Results saved!")