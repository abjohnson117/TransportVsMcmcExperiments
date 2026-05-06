import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm


n_projections = 2048
swd_seed = 42

# ---------------------------------------------------------------------------
# JAX sliced Wasserstein distance (memory-efficient, supports unequal sizes)
# ---------------------------------------------------------------------------
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
    t   = jnp.sort(jnp.concatenate([t_x, t_y]))

    @jax.jit
    def chunk_cost(key):
        theta = jax.random.normal(key, (chunk_size, d))
        theta = theta / jnp.linalg.norm(theta, axis=1, keepdims=True)
        xp = jnp.sort(x @ theta.T, axis=0).T
        yp = jnp.sort(y @ theta.T, axis=0).T

        def one_cost(xi, yi):
            return jnp.mean((jnp.interp(t, t_x, xi) - jnp.interp(t, t_y, yi)) ** p)

        return jnp.mean(jax.vmap(one_cost)(xp, yp))

    keys = jax.random.split(jax.random.PRNGKey(seed), n_projections // chunk_size)
    cost = jnp.mean(jnp.array([chunk_cost(k) for k in keys]))
    return cost ** (1.0 / p)


def main():
    train_dim   = 100000
    nsamples    = 40000
    nx = ny     = 33
    flat_length = nx * ny
    RESHAPE_NO  = 200000
    THIN_FACTOR = 10
    CHAIN_ITERS = 2

    # -----------------------------------------------------------------------
    # Load h-MALA reference chains (main observation only)
    # -----------------------------------------------------------------------
    print("Loading h-MALA chains (main observation)...")
    chains = []
    for i in tqdm(range(CHAIN_ITERS), desc="chains (obs)"):
        hmala_path = os.path.join("mcmc_main_ref", f"chain_{i:03d}", "hmala_samples.npz")
        chain = np.load(hmala_path)["arr"].reshape(RESHAPE_NO, flat_length)
        chains.append(chain[::THIN_FACTOR, :])
    hmala_samps = np.vstack(chains)
    print(f"h-MALA (obs) shape: {hmala_samps.shape}")

    # -----------------------------------------------------------------------
    # Load training data and compute base SWD (prior vs. h-MALA posterior)
    # -----------------------------------------------------------------------
    print("Loading training data...")
    us_all = np.load("training_dataset/parameters_noise.npy").reshape(-1, flat_length)

    us_ref = us_all[train_dim : train_dim * 2].copy()
    np.random.shuffle(us_ref)
    rng_base = np.random.default_rng(swd_seed)
    ref_idxs = rng_base.choice(len(us_ref), size=nsamples, replace=False)
    us_ref_subset = jnp.asarray(us_ref[ref_idxs])

    base_swd = float(sliced_wasserstein_jax(
        us_ref_subset,
        jnp.asarray(hmala_samps),
        n_projections=n_projections,
        seed=swd_seed,
    ))
    print(f"  Base SWD (obs): {base_swd:.6f}")

    # -----------------------------------------------------------------------
    # Sample-size list
    # -----------------------------------------------------------------------
    sample_no_list = [2 ** i for i in range(1, 15)]
    sample_no_list += [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, train_dim]
    sample_no_list = sorted(set(sample_no_list))

    swd_obs = np.empty(len(sample_no_list))

    # -----------------------------------------------------------------------
    # Single trial: chain_000
    # -----------------------------------------------------------------------
    mcmc_obs = np.load("pcn_main/chain_000/pcn_samples.npy").reshape(-1, flat_length)

    for j, sample_no in enumerate(tqdm(sample_no_list)):
        swd_obs[j] = float(sliced_wasserstein_jax(
            jnp.asarray(mcmc_obs[:sample_no]),
            jnp.asarray(hmala_samps),
            n_projections=n_projections,
            seed=swd_seed,
        )) / base_swd

    output_root = "convergence_results_final"
    np.save(os.path.join(output_root, "pcn_swd_obs_single.npy"), swd_obs)
    np.save(os.path.join(output_root, "pcn_base_swd_obs.npy"), np.array([base_swd]))
    print("Results saved.")


if __name__ == "__main__":
    main()
