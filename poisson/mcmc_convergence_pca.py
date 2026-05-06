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


# ---------------------------------------------------------------------------
# Whitened PCA
# ---------------------------------------------------------------------------
def get_pca_fns(us, explained_var_threshold=0.98):
    n = us.shape[0]
    mean_us = us.mean(axis=0)
    X = us - mean_us

    _, S, Vt = np.linalg.svd(X / np.sqrt(n - 1), full_matrices=False)
    V = Vt.T

    expl_var = (S ** 2) / (S ** 2).sum()
    k = int(np.searchsorted(np.cumsum(expl_var), explained_var_threshold) + 1)

    V_kept, S_kept = V[:, :k], S[:k]

    def pca_encode(b):
        return (b - mean_us) @ V_kept / S_kept

    def pca_decode(z):
        return mean_us + (z * S_kept) @ V_kept.T

    return pca_encode, pca_decode, k


def main():
    train_dim   = 100000
    nsamples    = 40000
    nx = ny     = 33
    flat_length = nx * ny
    RESHAPE_NO  = 200000
    THIN_FACTOR = 10
    CHAIN_ITERS = 2
    AUX_RESHAPE_NO  = 200000
    AUX_THIN_FACTOR = 10
    AUX_CHAIN_ITERS = 2
    COND_LABELS = ["obs", "med", "98"]

    # -----------------------------------------------------------------------
    # Load h-MALA reference chains
    # -----------------------------------------------------------------------
    print("Loading h-MALA chains (main observation)...")
    chains = []
    for i in tqdm(range(CHAIN_ITERS), desc="chains (obs)"):
        hmala_path = os.path.join("mcmc_main_ref", f"chain_{i:03d}", "hmala_samples.npz")
        chain = np.load(hmala_path)["arr"].reshape(RESHAPE_NO, flat_length)
        chains.append(chain[::THIN_FACTOR, :])
    hmala_samps = np.vstack(chains)
    print(f"h-MALA (obs) shape: {hmala_samps.shape}")

    print("Loading h-MALA chains (median observation)...")
    chains = []
    for i in tqdm(range(AUX_CHAIN_ITERS), desc="chains (med)"):
        hmala_path = os.path.join("mcmc_med_ref", f"chain_{i:03d}", "hmala_samples.npz")
        chain = np.load(hmala_path)["arr"].reshape(AUX_RESHAPE_NO, flat_length)
        chains.append(chain[::AUX_THIN_FACTOR, :])
    hmala_med = np.vstack(chains)
    print(f"h-MALA (med) shape: {hmala_med.shape}")

    print("Loading h-MALA chains (98th-pct observation)...")
    chains = []
    for i in tqdm(range(AUX_CHAIN_ITERS), desc="chains (98)"):
        hmala_path = os.path.join("mcmc_98_ref", f"chain_{i:03d}", "hmala_samples.npz")
        chain = np.load(hmala_path)["arr"].reshape(AUX_RESHAPE_NO, flat_length)
        chains.append(chain[::AUX_THIN_FACTOR, :])
    hmala_98 = np.vstack(chains)
    print(f"h-MALA (98)  shape: {hmala_98.shape}")

    # -----------------------------------------------------------------------
    # Load training data and fit PCA on reference split
    # -----------------------------------------------------------------------
    print("Loading training data...")
    us_all = np.load("training_dataset/parameters_noise.npy").reshape(-1, flat_length)

    us_ref = us_all[train_dim : train_dim * 2].copy()
    np.random.shuffle(us_ref)

    print("Computing PCA on reference parameters...")
    pca_encode, pca_decode, k = get_pca_fns(us_ref)
    print(f"PCA retains {k} components (98% variance explained)")

    # Encode h-MALA reference chains into PCA space
    hmala_pca     = jnp.asarray(pca_encode(hmala_samps))
    hmala_med_pca = jnp.asarray(pca_encode(hmala_med))
    hmala_98_pca  = jnp.asarray(pca_encode(hmala_98))
    hmala_pca_list = [hmala_pca, hmala_med_pca, hmala_98_pca]

    # -----------------------------------------------------------------------
    # Base SWD in PCA space (prior vs. each h-MALA posterior)
    # -----------------------------------------------------------------------
    us_ref_pca = jnp.asarray(pca_encode(us_ref))
    rng_base = np.random.default_rng(swd_seed)
    ref_idxs = rng_base.choice(len(us_ref_pca), size=nsamples, replace=False)
    us_ref_pca_subset = us_ref_pca[ref_idxs]

    base_swd_list = []
    for label, h_pca in zip(COND_LABELS, hmala_pca_list):
        bswd = float(sliced_wasserstein_jax(
            us_ref_pca_subset,
            h_pca,
            n_projections=n_projections,
            seed=swd_seed,
        ))
        base_swd_list.append(bswd)
        print(f"  Base SWD ({label}, PCA): {bswd:.6f}")

    # -----------------------------------------------------------------------
    # Sample-size list
    # -----------------------------------------------------------------------
    sample_no_list = [2 ** i for i in range(1, 15)]
    sample_no_list += [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, train_dim]
    sample_no_list = sorted(set(sample_no_list))

    swd_obs = np.empty(len(sample_no_list))
    swd_med = np.empty(len(sample_no_list))
    swd_98  = np.empty(len(sample_no_list))

    # -----------------------------------------------------------------------
    # Single trial: load convergence chains, encode to PCA, compute SWD
    # -----------------------------------------------------------------------
    mcmc_obs = np.load("mcmc_main_converge_samps/chain_000/hmala_samples.npz")["arr"].reshape(-1, flat_length)
    mcmc_med = np.load("mcmc_med_converge_samps/chain_000/hmala_samples.npz")["arr"].reshape(-1, flat_length)
    mcmc_98  = np.load("mcmc_98_converge_samps/chain_000/hmala_samples.npz")["arr"].reshape(-1, flat_length)

    for j, sample_no in enumerate(tqdm(sample_no_list)):
        obs_pca = jnp.asarray(pca_encode(mcmc_obs[:sample_no]))
        med_pca = jnp.asarray(pca_encode(mcmc_med[:sample_no]))
        p98_pca = jnp.asarray(pca_encode(mcmc_98[:sample_no]))

        swd_obs[j] = float(sliced_wasserstein_jax(
            obs_pca, hmala_pca, n_projections=n_projections, seed=swd_seed,
        )) / base_swd_list[0]
        swd_med[j] = float(sliced_wasserstein_jax(
            med_pca, hmala_med_pca, n_projections=n_projections, seed=swd_seed,
        )) / base_swd_list[1]
        swd_98[j] = float(sliced_wasserstein_jax(
            p98_pca, hmala_98_pca, n_projections=n_projections, seed=swd_seed,
        )) / base_swd_list[2]

    output_root = "convergence_results_final"
    np.save(os.path.join(output_root, "mcmc_swd_pca_obs.npy"), swd_obs)
    np.save(os.path.join(output_root, "mcmc_swd_pca_med.npy"), swd_med)
    np.save(os.path.join(output_root, "mcmc_swd_pca_98.npy"),  swd_98)
    np.save(os.path.join(output_root, "mcmc_base_swd_pca.npy"), np.array(base_swd_list))
    print("Results saved.")


if __name__ == "__main__":
    main()
