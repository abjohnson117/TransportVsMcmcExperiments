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


def load_hmala_chains(folder, no_samples, thin, n_chains=3):
    chains = []
    for i in range(n_chains):
        c = np.load(os.path.join(folder, f"chain_{i:03d}", "hmala_samples_delta.npy"))
        chains.append(c.reshape(no_samples, 21, 21, 3, order="F")[::thin, :, :, 0])
    return np.concatenate(chains).reshape(-1, flat_length)


def load_mala_chain(folder):
    """Load single MALA chain; returns z=0 surface, shape (N, 441)."""
    raw = np.load(os.path.join(folder, "chain_000", "mala_samples.npy"))
    N = raw.shape[0]
    return raw.reshape(N, 21, 21, 3, order="F")[:, :, :, 0].reshape(N, flat_length)


def main():
    HMALA_NO_SAMPLES = 300_000
    HMALA_THIN       = 20
    train_dim        = 100_000
    nsamples         = 50_000

    print("Loading h-MALA reference chain...")
    hmala_ref = load_hmala_chains("mcmc_main", HMALA_NO_SAMPLES, HMALA_THIN)
    print(f"h-MALA reference shape: {hmala_ref.shape}")

    print("Loading MALA chain...")
    mala_chain = load_mala_chain("mcmc_main_mala")
    N = mala_chain.shape[0]
    print(f"MALA chain shape: {mala_chain.shape}")

    print("Loading reference parameters for base SWD computation...")
    us_all = (np.load("training_dataset/parameters_delta.npy")
              .reshape(-1, 21, 21, 3, order="F")[:, :, :, 0]
              .reshape(-1, flat_length))
    us_ref = us_all[train_dim : train_dim * 2].copy()

    rng_base = np.random.default_rng(swd_seed)
    ref_idxs = rng_base.choice(len(us_ref), size=nsamples, replace=False)
    us_ref_subset = us_ref[ref_idxs]

    print("Computing base SWD (prior vs. h-MALA reference)...")
    base_swd = float(sliced_wasserstein_jax(
        jnp.asarray(us_ref_subset),
        jnp.asarray(hmala_ref),
        n_projections=n_projections,
        seed=swd_seed,
    ))
    print(f"Base SWD: {base_swd:.6f}")

    # Powers of 2 up to N, plus a few evenly-spaced points near the end
    sample_no_list = [2 ** i for i in range(1, 20) if 2 ** i <= N]
    sample_no_list += [int(N * f) for f in [0.4, 0.6, 0.8, 1.0]]
    sample_no_list = sorted(set(n for n in sample_no_list if 1 <= n <= N))
    print(f"Sample sizes: {sample_no_list}")

    swds = np.empty(len(sample_no_list))
    for j, sample_no in enumerate(tqdm(sample_no_list, desc="SWD over prefixes")):
        swds[j] = float(sliced_wasserstein_jax(
            jnp.asarray(mala_chain[:sample_no]),
            jnp.asarray(hmala_ref),
            n_projections=n_projections,
            seed=swd_seed,
        ))

    rel_swds = swds / base_swd

    output_root = "convergence_results_mala"
    os.makedirs(output_root, exist_ok=True)
    np.save(os.path.join(output_root, "sample_no_list.npy"), np.array(sample_no_list))
    np.save(os.path.join(output_root, "swd.npy"),            swds)
    np.save(os.path.join(output_root, "rel_swd.npy"),        rel_swds)
    np.save(os.path.join(output_root, "base_swd.npy"),       np.array([base_swd]))
    print(f"Results saved to {output_root}/. Done.")


if __name__ == "__main__":
    main()
