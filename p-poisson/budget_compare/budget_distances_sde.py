import os
import sys
import argparse

# Must be set before JAX initializes its backend
_argv = sys.argv
_gpu = next((int(_argv[i + 1]) for i, a in enumerate(_argv[:-1]) if a == "--gpu_idx"), 0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",      type=int, default=0,             help="Run index or ID for output folder")
    parser.add_argument("--ref_root",    type=str, default="../mcmc_256", help="Root directory containing reference chain_* folders")
    parser.add_argument("--gpu_idx",     type=int, default=0,             help="Index of GPU to use")
    parser.add_argument("--chain_start", type=int, default=0,             help="First chain index (inclusive)")
    parser.add_argument("--chain_end",   type=int, default=865,           help="Last chain index (exclusive)")
    args = parser.parse_args()
    run_id      = args.run_id
    chain_start = args.chain_start
    chain_end   = args.chain_end

    output_dir = f"budget_results{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    swd_seed      = 42
    n_projections = 2048

    abs_indices  = list(range(chain_start, chain_end))
    n_cond_vars  = len(abs_indices)
    swd_nn_array = np.empty(n_cond_vars)

    for local_idx, abs_idx in enumerate(tqdm(abs_indices)):
        # Reference: h-MALA samples for this chain — shape (n_mcmc, flat_length)
        ref_path  = os.path.join(args.ref_root, f"chain_{abs_idx:03d}", "hmala_samples.npy")
        ref_samps = np.load(ref_path).reshape(-1, 21, 21, 3, order="F")[:, :, :, 0].reshape(-1, 21 ** 2)   # (n_mcmc, 441) z=0 layer

        # Transport: SI samples for this chain — shape (nsamples, flat_length)
        nn_path  = os.path.join(
            "..", "budget_transport", "budget_transport_samps",
            f"run_{run_id:02d}", f"si_sde_samps_{abs_idx:03d}.npy"
        )
        nn_samps = np.load(nn_path)     # (nsamples, flat_length)

        swd_nn_array[local_idx] = float(sliced_wasserstein_jax(
            jnp.asarray(nn_samps),
            jnp.asarray(ref_samps),
            n_projections=n_projections,
            seed=swd_seed,
        ))
        if local_idx % 20 == 0:
            print(f"This is the swd: {swd_nn_array[local_idx]}")

    np.save(os.path.join(output_dir, f"swd_nn_array_{run_id}.npy"), swd_nn_array)


if __name__ == "__main__":
    main()
