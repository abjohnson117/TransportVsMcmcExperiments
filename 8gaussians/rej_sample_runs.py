import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
from jax import vmap, grad, jit, random, lax
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import minimize_scalar
import argparse
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()

def make_centers(scale=4.0, dtype=np.float64):
    base = np.array([
        [ 1.0,  0.0],
        [-1.0,  0.0],
        [ 0.0,  1.0],
        [ 0.0, -1.0],
        [ 1.0/np.sqrt(2.0),  1.0/np.sqrt(2.0)],
        [ 1.0/np.sqrt(2.0), -1.0/np.sqrt(2.0)],
        [-1.0/np.sqrt(2.0),  1.0/np.sqrt(2.0)],
        [-1.0/np.sqrt(2.0), -1.0/np.sqrt(2.0)],
    ], dtype=dtype)
    centers = (scale / np.sqrt(2.0)) * base  # (8,2)
    return centers

CENTERS = make_centers(scale=4.0, dtype=np.float64)
FACTOR = 1.0 / np.sqrt(np.pi)  # your factor

def rejection_sample(n, M, y, tau=5.0, batch=4096, seed=0):
    rng = np.random.default_rng(seed)
    log_m = np.log(M)

    out = np.empty(n, dtype=np.float64)
    filled = 0
    total = 0

    while filled < n:
        u_prop = rng.normal(loc=0.0, scale=tau, size=batch)

        log_alpha = (
            log_eight_gaussians_cond(u_prop, y, scale=4.0)
            - log_gauss(u_prop, tau)
            - log_m
        )

        alpha = np.exp(np.minimum(log_alpha, 0.0))
        accept = rng.uniform(size=batch) < alpha
        accepted = u_prop[accept]

        k = min(accepted.size, n - filled)
        if k > 0:
            out[filled:filled+k] = accepted[:k]
            filled += k

        total += batch

    return out, n / total

output_root = "rej_results"
output_dir = os.path.join(output_root, f"_{args.run_id}")
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
nsamples = 10000
tau = 5.0
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-3.1, high=3.1, size=(budget, ))

rej_samps = np.zeros((budget, nsamples))

for i, y in enumerate(tqdm(conditioning_ys)):
    def log_eight_gaussians_cond(u, y, scale=4.0, centers=CENTERS, log_factor=np.log(FACTOR)):
        u = np.asarray(u, dtype=np.float64)
        y = float(y)

        x0 = np.full_like(u, y, dtype=np.float64)
        x = np.stack([x0, u], axis=-1)                 # (...,2)

        diffs = x[..., None, :] - centers[None, :, :]  # (...,8,2)
        sqdist = np.sum(diffs * diffs, axis=-1)        # (...,8)

        a = -scale * sqdist                            # (...,8)
        m = np.max(a, axis=-1, keepdims=True)          # (...,1)
        lse = np.log(np.sum(np.exp(a - m), axis=-1)) + m.squeeze(-1)  # (...,)
        return log_factor + lse

    def log_gauss(u, tau):
        u = np.asarray(u, dtype=np.float64)
        return -0.5*(u*u)/(tau*tau) - 0.5*np.log(2*np.pi*tau*tau)
    
    def neg_log_ratio(u):
        return -(log_eight_gaussians_cond(u, y) - log_gauss(u, tau))
    
    res = minimize_scalar(neg_log_ratio, bounds=(-10, 10), method="bounded")
    u_star = res.x
    max_log_ratio = -res.fun
    M = np.exp(max_log_ratio)

    samps, accept_rate = rejection_sample(nsamples, M=M, y=y, tau=tau, seed=i)
    rej_samps[i, :] = samps

print("Saving results...")
np.save(os.path.join(output_dir, "conditioning_ys.npy"), conditioning_ys)
np.save(os.path.join(output_dir, "rej_samps.npy"), rej_samps)
print("Results saved!")