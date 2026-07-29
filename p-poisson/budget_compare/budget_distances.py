import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_id",      type=int, default=0,             help="Run index or ID for output folder")
parser.add_argument("--ref_root",    type=str, default="../mcmc_256", help="Root directory containing reference chain_* folders and budget_results")
parser.add_argument("--gpu_idx",     type=int, default=1,             help="Index of GPU to use")
parser.add_argument("--chain_start", type=int, default=0,             help="First chain index (inclusive)")
parser.add_argument("--chain_end",   type=int, default=865,           help="Last chain index (exclusive)")
args = parser.parse_args()
run_id      = args.run_id
GPU_IDX     = args.gpu_idx
chain_start = args.chain_start
chain_end   = args.chain_end

os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_IDX}"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


def sliced_wasserstein_jax(x, y, n_projections=512, seed=42, chunk_size=64, p=2):
    """
    Memory-efficient SWD in JAX. Processes projections in chunks.
    Supports unequal sample sizes via quantile interpolation.
    chunk_size=64 uses ~100 MB/kernel; increase for speed, decrease for memory.
    """
    n_x, d = x.shape
    n_y = y.shape[0]
    assert n_projections % chunk_size == 0

    t_x = (jnp.arange(n_x, dtype=jnp.float32) + 0.5) / n_x
    t_y = (jnp.arange(n_y, dtype=jnp.float32) + 0.5) / n_y
    t   = jnp.sort(jnp.concatenate([t_x, t_y]))  # common quantile grid

    @jax.jit
    def chunk_cost(key):
        theta = jax.random.normal(key, (chunk_size, d))
        theta = theta / jnp.linalg.norm(theta, axis=1, keepdims=True)
        xp = jnp.sort(x @ theta.T, axis=0).T   # (chunk_size, n_x)
        yp = jnp.sort(y @ theta.T, axis=0).T   # (chunk_size, n_y)

        def one_cost(xi, yi):
            return jnp.mean((jnp.interp(t, t_x, xi) - jnp.interp(t, t_y, yi)) ** p)

        return jnp.mean(jax.vmap(one_cost)(xp, yp))

    keys = jax.random.split(jax.random.PRNGKey(seed), n_projections // chunk_size)
    cost = jnp.mean(jnp.array([chunk_cost(k) for k in keys]))
    return cost ** (1.0 / p)

output_dir = f"budget_results{run_id}"
os.makedirs(output_dir, exist_ok=True)

seed     = run_id
swd_seed = 42
n_projections = 2048

# Budget parameters — matching budget_mcmc_runs.py
# nfevs=256 MAP cost; each h-MALA step = 2 PDE solves → divide by 2
conditioning_list = [4 ** i for i in range(5)]          # [1, 4, 16, 64, 256]
chain_list        = [8064, 1920, 384, 1, 1]             # steps per cond var

# All y_obs vectors in [chain_start, chain_end) — used to match cond_vars back to chain indices
N_CHAINS = chain_end - chain_start
all_y_obs = np.array([
    np.load(os.path.join(args.ref_root, f"chain_{idx:03d}", "y_obs.npy"))
    for idx in range(chain_start, chain_end)
])  # (N_CHAINS, n_obs)

wd_mcmc_array     = np.zeros(len(conditioning_list))
wd_mcmc_array_std = np.zeros(len(conditioning_list))

for i, cond_num in enumerate(tqdm(conditioning_list)):
    chain_length = chain_list[i]

    # cond_vars: (cond_num, n_obs) — subset of y_obs used at this budget level
    cond_vars = np.load(
        os.path.join(args.ref_root, f"budget_results_{run_id}", f"cond_vars_{cond_num}.npy")
    )
    # mcmc_samps: (cond_num, chain_length, num_vertices)
    mcmc_samps = np.load(
        os.path.join(args.ref_root, f"budget_results_{run_id}",
                     f"mcmc_samps_{chain_length}_{cond_num}.npy")
    )
    if mcmc_samps.ndim == 2:
        mcmc_samps = np.expand_dims(mcmc_samps, axis=0)

    # Match each cond_var back to its absolute chain index
    local_idxs = np.array([
        np.where((all_y_obs == cv).all(axis=1))[0][0] for cv in cond_vars
    ])
    abs_idxs = local_idxs + chain_start

    # Filter to chains that actually have reference samples
    valid_mask = np.array([
        os.path.exists(os.path.join(args.ref_root, f"chain_{idx:03d}", "hmala_samples.npy"))
        for idx in abs_idxs
    ])
    if not valid_mask.all():
        print(f"  [cond_num={cond_num}] skipping {(~valid_mask).sum()} chains without hmala_samples.npy")
    abs_idxs  = abs_idxs[valid_mask]
    mcmc_samps = mcmc_samps[valid_mask]

    mcmc_array = np.zeros(len(abs_idxs))
    for j in range(len(abs_idxs)):
        ref_samps = np.load(
            os.path.join(args.ref_root, f"chain_{abs_idxs[j]:03d}", "hmala_samples.npy")
        )  # (n_mcmc, num_vertices)

        # Project both to the z=0 bottom layer (441-dim, Fortran order) to match
        # the space in which SI transport samples live.
        mcmc_j_z0 = mcmc_samps[j].reshape(-1, 21, 21, 3, order="F")[:, :, :, 0].reshape(-1, 441)
        ref_z0    = ref_samps.reshape(-1, 21, 21, 3, order="F")[:, :, :, 0].reshape(-1, 441)

        mcmc_array[j] = float(sliced_wasserstein_jax(
            jnp.asarray(mcmc_j_z0),
            jnp.asarray(ref_z0),
            n_projections=n_projections,
            seed=swd_seed,
        ))

        if j % 20 == 0:
            print(f"  SWD for MCMC [{j}]: {mcmc_array[j]}")

    wd_mcmc_array[i]     = np.mean(mcmc_array)
    wd_mcmc_array_std[i] = np.std(mcmc_array)

np.save(os.path.join(output_dir, "swd_mcmc_array_mean.npy"), wd_mcmc_array)
np.save(os.path.join(output_dir, "swd_mcmc_array_std.npy"),  wd_mcmc_array_std)
