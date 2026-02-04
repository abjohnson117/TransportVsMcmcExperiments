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

@partial(jit, static_argnames=("n", "batch"))
def rejection_sample(n: int, M: float, tau: float = 5.0, batch: int = 4096, seed: int = 0):
    key = random.PRNGKey(seed)
    log_m = jnp.log(M)

    log_target = vmap(log_eight_gaussians_cond)
    log_prop   = vmap(log_gauss)

    # Buffer with extra space so we can always write `batch` samples safely.
    out = jnp.zeros((n + batch,), dtype=jnp.float32)
    filled = jnp.array(0, dtype=jnp.int32)
    total  = jnp.array(0, dtype=jnp.int32)

    def cond_fn(state):
        key, out, filled, total = state
        return filled < n

    def body_fn(state):
        key, out, filled, total = state
        key, k1, k2 = random.split(key, 3)

        u_prop = random.normal(k1, shape=(batch,)) * tau

        log_alpha = log_target(u_prop) - log_prop(u_prop) - log_m
        alpha = jnp.exp(jnp.minimum(log_alpha, 0.0))

        u = random.uniform(k2, shape=(batch,))
        accept = u < alpha

        # Static-shape "packing" of accepted proposals:
        # idx has shape (batch,), with -1 fill for rejected slots
        idx = jnp.nonzero(accept, size=batch, fill_value=-1)[0]
        packed = jnp.where(idx >= 0, u_prop[idx], 0.0)  # shape (batch,)

        # Always write `batch` values (safe because out has length n+batch)
        out = lax.dynamic_update_slice(out, packed, (filled,))
        filled = filled + accept.sum().astype(jnp.int32)
        total  = total + batch

        return (key, out, filled, total)

    key, out, filled, total = lax.while_loop(cond_fn, body_fn, (key, out, filled, total))

    samples = out[:n]
    accept_rate = n / total.astype(jnp.float32)
    return samples, accept_rate


output_root = "rej_results"
output_dir = os.path.join(output_root, f"_{args.run_id}")
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
nsamples = 50000
tau = 5.0
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-3.1, high=3.1, size=(budget, ))

rej_samps = np.zeros((budget, nsamples))

for i, y in enumerate(tqdm(conditioning_ys)):
    def make_eight_gaussians_cond(y, scale=4.0):
    # Precompute centers as a JAX array (constants)
        base = jnp.array([
            [ 1.0,  0.0],
            [-1.0,  0.0],
            [ 0.0,  1.0],
            [ 0.0, -1.0],
            [ 1.0/jnp.sqrt(2.0),  1.0/jnp.sqrt(2.0)],
            [ 1.0/jnp.sqrt(2.0), -1.0/jnp.sqrt(2.0)],
            [-1.0/jnp.sqrt(2.0),  1.0/jnp.sqrt(2.0)],
            [-1.0/jnp.sqrt(2.0), -1.0/jnp.sqrt(2.0)],
        ])  # (8,2)

        centers = (scale / jnp.sqrt(2.0)) * base  # matches your construction

        factor = 1.0 / jnp.sqrt(jnp.pi)

        def eight_gaussians_cond(u):
            # u can be scalar or shape (batch,)
            u = jnp.asarray(u)
            x = jnp.stack([jnp.full_like(u, y), u], axis=-1)   # (..., 2)

            # squared distances to each center, broadcasted
            # x[..., None, :] -> (..., 1, 2); centers[None, ...] -> (1, 8, 2)
            diffs = x[..., None, :] - centers[None, :, :]      # (..., 8, 2)
            norm_part = jnp.sum(diffs * diffs, axis=-1)        # (..., 8)

            exp_part = jnp.exp(-scale * norm_part)             # (..., 8)
            return factor * jnp.sum(exp_part, axis=-1)         # (...)

        def log_eight_gaussians_cond(u):
            # if you ever hit underflow, you can add tiny eps:
            return jnp.log(eight_gaussians_cond(u) + 1e-30)

        return eight_gaussians_cond, log_eight_gaussians_cond
    
    eight_gaussians_cond, log_eight_gaussians_cond = make_eight_gaussians_cond(y=y)

    def log_gauss(u):
        # u = float(u)
        # IMPORTANT: include normalization if you want M for rejection sampling
        return -0.5*(u*u)/(tau*tau) - 0.5*np.log(2*np.pi*tau*tau)

    def neg_log_ratio(u):
        return -(log_eight_gaussians_cond(u).item() - log_gauss(u).item())
    
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