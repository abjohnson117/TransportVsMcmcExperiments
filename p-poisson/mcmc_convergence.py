import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm


n_projections = 2048
swd_seed = 42
nx = ny = 20
flat_length = (nx + 1) * (ny + 1)  # 441

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


def load_hmala_chains(folder, no_samples, thin, is_delta=False, n_chains=3):
    """Load and concatenate h-MALA chains; returns z=0 surface, shape (n_total, 441)."""
    chains = []
    for i in range(n_chains):
        if is_delta:
            c = np.load(os.path.join(folder, f"chain_{i:03d}", "hmala_samples_delta.npy"))
        else:
            c = np.load(os.path.join(folder, f"chain_{i:03d}", "hmala_samples.npy"))
        chains.append(c.reshape(no_samples, 21, 21, 3, order="F")[::thin, :, :, 0])
    return np.concatenate(chains).reshape(-1, flat_length)


def main():
    ntrials = 10
    HMALA_NO_SAMPLES = 300_000
    HMALA_THIN       = 20
    COND_LABELS = ["obs", "med", "98"]
    train_dim   = 100_000
    nsamples    = 50_000

    print("Loading h-MALA chains...")
    hmala_samps = load_hmala_chains("mcmc_main",   HMALA_NO_SAMPLES, HMALA_THIN, is_delta=True)
    hmala_med   = load_hmala_chains("mcmc_median", HMALA_NO_SAMPLES, HMALA_THIN)
    hmala_98    = load_hmala_chains("mcmc_98",     HMALA_NO_SAMPLES, HMALA_THIN)
    print(f"h-MALA (obs): {hmala_samps.shape},  (med): {hmala_med.shape},  (98): {hmala_98.shape}")
    hmala_list = [hmala_samps, hmala_med, hmala_98]

    print("Loading reference parameters for base SWD computation...")
    us_all = (np.load("training_dataset/parameters_delta.npy")
              .reshape(-1, 21, 21, 3, order="F")[:, :, :, 0]
              .reshape(-1, flat_length))
    us_ref = us_all[train_dim : train_dim * 2].copy()

    rng_base = np.random.default_rng(swd_seed)
    ref_idxs = rng_base.choice(len(us_ref), size=nsamples, replace=False)
    us_ref_subset = us_ref[ref_idxs]

    print("Computing base SWDs (prior vs. each h-MALA set)...")
    base_swd_list = []
    for label, h_flat in zip(COND_LABELS, hmala_list):
        bswd = float(sliced_wasserstein_jax(
            jnp.asarray(us_ref_subset),
            jnp.asarray(h_flat),
            n_projections=n_projections,
            seed=swd_seed,
        ))
        base_swd_list.append(bswd)
        print(f"  Base SWD ({label}): {bswd:.6f}")

    sample_no_list = [2 ** i for i in range(1, 15)]          # 2 … 16384
    sample_no_list += [20000, 30000, 40000, 50000]
    sample_no_list  = sorted(set(sample_no_list))

    swd_obs = np.empty((ntrials, len(sample_no_list)))
    swd_med = np.empty((ntrials, len(sample_no_list)))
    swd_98  = np.empty((ntrials, len(sample_no_list)))

    for i in tqdm(range(ntrials)):
        def load_converge(folder, is_delta=False):
            if is_delta:
                c = np.load(f"{folder}/chain_{i:03d}/hmala_samples_delta.npy")
            else:
                c = np.load(f"{folder}/chain_{i:03d}/hmala_samples.npy")
            N = c.shape[0]
            return c.reshape(N, 21, 21, 3, order="F")[:, :, :, 0].reshape(N, flat_length)

        mcmc_obs = load_converge("mcmc_main_converge_samps", is_delta=True)
        mcmc_med = load_converge("mcmc_med_converge_samps")
        mcmc_98  = load_converge("mcmc_98_converge_samps")

        for j, sample_no in enumerate(tqdm(sample_no_list)):
            mcmc_obs_sub = mcmc_obs[:sample_no, :]
            mcmc_med_sub = mcmc_med[:sample_no, :]
            mcmc_98_sub  = mcmc_98[:sample_no, :]

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

    rel_swd_obs = swd_obs / base_swd_list[0]
    rel_swd_med = swd_med / base_swd_list[1]
    rel_swd_98  = swd_98  / base_swd_list[2]

    output_root = "convergence_results_final"
    os.makedirs(output_root, exist_ok=True)
    np.save(os.path.join(output_root, "sample_no_list.npy"), np.array(sample_no_list))
    np.save(os.path.join(output_root, "swd_obs.npy"), swd_obs)
    np.save(os.path.join(output_root, "swd_med.npy"), swd_med)
    np.save(os.path.join(output_root, "swd_98.npy"),  swd_98)
    np.save(os.path.join(output_root, "rel_swd_obs.npy"), rel_swd_obs)
    np.save(os.path.join(output_root, "rel_swd_med.npy"), rel_swd_med)
    np.save(os.path.join(output_root, "rel_swd_98.npy"),  rel_swd_98)
    print("Results saved. Done.")

if __name__ == "__main__":
    main()