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
    ntrials    = 10
    nsamples    = 40000
    nx = ny     = 33
    flat_length = nx * ny
    RESHAPE_NO  = 200000
    THIN_FACTOR = 10
    CHAIN_ITERS = 2
    # Auxiliary chains (mcmc_median/, mcmc_98/): 20 chains, 47,000 raw samples each.
    # Remove 7,000 burn-in → 40,000 usable; thin by 20 → 2,000/chain × 20 chains = 40,000.
    AUX_RESHAPE_NO  = 200000
    AUX_THIN_FACTOR = 10
    AUX_CHAIN_ITERS = 2
    COND_LABELS = ["obs", "med", "98"]
    print("Loading h-MALA chains (main observation)...")
    chains = []
    for i in tqdm(range(CHAIN_ITERS), desc="chains (obs)"):
        hmala_path = os.path.join("mcmc_main_ref", f"chain_{i:03d}", "hmala_samples.npz")
        chain = np.load(hmala_path)["arr"].reshape(RESHAPE_NO, flat_length)
        chains.append(chain[::THIN_FACTOR, :])
    hmala_samps = np.vstack(chains)  # (98_000, flat_length)
    print(f"h-MALA (obs) shape: {hmala_samps.shape}")

    # ---------------------------------------------------------------------------
    # Load h-MALA chains — median conditioning value
    # ---------------------------------------------------------------------------
    print("Loading h-MALA chains (median observation)...")
    chains = []
    for i in tqdm(range(AUX_CHAIN_ITERS), desc="chains (med)"):
        hmala_path = os.path.join("mcmc_med_ref", f"chain_{i:03d}", "hmala_samples.npz")
        chain = np.load(hmala_path)["arr"].reshape(AUX_RESHAPE_NO, flat_length)
        chains.append(chain[::AUX_THIN_FACTOR, :])
    hmala_med = np.vstack(chains)  # (98_000, flat_length)
    print(f"h-MALA (med) shape: {hmala_med.shape}")

    # ---------------------------------------------------------------------------
    # Load h-MALA chains — 98th-percentile conditioning value
    # ---------------------------------------------------------------------------
    print("Loading h-MALA chains (98th-pct observation)...")
    chains = []
    for i in tqdm(range(AUX_CHAIN_ITERS), desc="chains (98)"):
        hmala_path = os.path.join("mcmc_98_ref", f"chain_{i:03d}", "hmala_samples.npz")
        chain = np.load(hmala_path)["arr"].reshape(AUX_RESHAPE_NO, flat_length)
        chains.append(chain[::AUX_THIN_FACTOR, :])
    hmala_98 = np.vstack(chains)  # (98_000, flat_length)
    print(f"h-MALA (98)  shape: {hmala_98.shape}")

    hmala_list = [hmala_samps, hmala_med, hmala_98]

    print("Loading training data...")
    ys_all = np.load("training_dataset/solutions_grid_noise.npy")
    us_all = np.load("training_dataset/parameters_noise.npy").reshape(-1, flat_length)
    print(f"Full dataset: ys={ys_all.shape}, us={us_all.shape}")

    # Reference split: next train_dim samples (for PCA fitting and reference measure)
    us_ref = us_all[train_dim : train_dim * 2].copy()
    np.random.shuffle(us_ref)
    rng_base = np.random.default_rng(swd_seed)
    ref_idxs = rng_base.choice(len(us_ref), size=nsamples, replace=False)
    us_ref_subset = jnp.asarray(us_ref[ref_idxs])
    base_swd_list = []
    for label, hmala_flat in zip(COND_LABELS, hmala_list):
        bswd = float(sliced_wasserstein_jax(
            us_ref_subset,
            jnp.asarray(hmala_flat),
            n_projections=n_projections,
            seed=swd_seed,
        ))
        base_swd_list.append(bswd)
        print(f"  Base SWD ({label}): {bswd:.6f}")

    sample_no_list = [2 ** i for i in range(1, 15)]          # 2 … 16384
    sample_no_list += [20000, 30000, 40000, 50000,
                    60000, 70000, 80000, 90000, train_dim]
    sample_no_list  = sorted(set(sample_no_list))
    swd_obs = np.empty((ntrials, len(sample_no_list)))
    swd_med = np.empty((ntrials, len(sample_no_list)))
    swd_98 = np.empty((ntrials, len(sample_no_list)))

    for i in tqdm(range(ntrials)):
        mcmc_obs = np.load(f"mcmc_main_converge_samps/chain_{i:03d}/hmala_samples.npz")["arr"].reshape(-1, flat_length)
        mcmc_med = np.load(f"mcmc_med_converge_samps/chain_{i:03d}/hmala_samples.npz")["arr"].reshape(-1, flat_length)
        mcmc_98 = np.load(f"mcmc_98_converge_samps/chain_{i:03d}/hmala_samples.npz")["arr"].reshape(-1, flat_length)

        for j, sample_no in enumerate(tqdm(sample_no_list)):
            mcmc_obs_sub = mcmc_obs[:sample_no, :]
            mcmc_med_sub = mcmc_med[:sample_no, :]
            mcmc_98_sub = mcmc_98[:sample_no, :]

            swd_obs[i, j] = float(sliced_wasserstein_jax(
                jnp.asarray(mcmc_obs_sub),
                jnp.asarray(hmala_samps),
                n_projections=n_projections,
                seed=swd_seed,
            ))

            swd_med[i, j] = float(sliced_wasserstein_jax(
                jnp.asarray(mcmc_med_sub),
                jnp.asarray(hmala_med),
                n_projections=n_projections,
                seed=swd_seed,
            ))

            swd_98[i, j] = float(sliced_wasserstein_jax(
                jnp.asarray(mcmc_98_sub),
                jnp.asarray(hmala_98),
                n_projections=n_projections,
                seed=swd_seed,
            ))

    output_root = "convergence_results_final"
    np.save(os.path.join(output_root, "swd_obs.npy"), swd_obs)
    np.save(os.path.join(output_root, "swd_med.npy"), swd_med)
    np.save(os.path.join(output_root, "swd_98.npy"), swd_98)

if __name__ == "__main__":
    main()