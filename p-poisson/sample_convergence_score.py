import gc
import os
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import diffrax
import wandb
import equinox as eqx
from jax import grad, vmap, random
from tqdm.auto import tqdm
from typing import Callable, List

from triangular_transport.flows.flow_trainer import NNTrainer
from triangular_transport.flows.interpolants import (
    trig_interpolant,
    trig_interpolant_der,
)
from triangular_transport.flows.loss_functions import vec_field_loss
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer
from triangular_transport.flows.dataloaders import gaussian_reference_sampler

from conditional_sampling_sde import conditional_sample


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


# ---------------------------------------------------------------------------
# MLP with residual connections
# ---------------------------------------------------------------------------
class MLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    skips: List[eqx.nn.Linear | None]
    out: eqx.nn.Linear
    activation_fn: List[Callable]

    def __init__(
        self,
        key: jax.random.PRNGKey,
        dim: int,
        out_dim: int | None = None,
        num_layers: int = 4,
        activation_fn: List[Callable] | Callable = jax.nn.gelu,
        w: int | List[int] = 64,
        time_varying: bool = False,
    ):
        if out_dim is None:
            out_dim = dim

        if isinstance(activation_fn, list):
            if len(activation_fn) == 1:
                activation_fn *= num_layers - 1
        else:
            activation_fn = [activation_fn] * (num_layers - 1)
        self.activation_fn = activation_fn

        if isinstance(w, list):
            if len(w) == 1:
                w *= num_layers - 1
            widths = w
        else:
            widths = [w] * (num_layers - 1)

        k_keys = jax.random.split(key, 2 * num_layers)
        in_dim0 = dim + (1 if time_varying else 0)

        self.layers = []
        self.skips = []
        in_dim = in_dim0
        for i, width in enumerate(widths):
            self.layers.append(eqx.nn.Linear(in_dim, width, key=k_keys[i]))
            if in_dim == width:
                self.skips.append(None)
            else:
                self.skips.append(eqx.nn.Linear(in_dim, width, key=k_keys[i + num_layers]))
            in_dim = width

        self.out = eqx.nn.Linear(in_dim, out_dim, key=k_keys[-1])

    def __call__(self, x):
        for layer, skip, act in zip(self.layers, self.skips, self.activation_fn):
            h = act(layer(x))
            s = x if skip is None else skip(x)
            x = h + s
        return self.out(x)


# ---------------------------------------------------------------------------
# Whitened PCA with per-sample residual variance injection
# ---------------------------------------------------------------------------
def get_pca_fns(us, explained_var_threshold=0.99):
    n = us.shape[0]
    mean_us = us.mean(axis=0)
    X = us - mean_us

    _, S, Vt = np.linalg.svd(X / np.sqrt(n - 1), full_matrices=False)
    V = Vt.T

    expl_var = (S ** 2) / (S ** 2).sum()
    k = int(np.searchsorted(np.cumsum(expl_var), explained_var_threshold) + 1)

    V_kept, S_kept = V[:, :k], S[:k]
    V_res,  S_res  = V[:, k:], S[k:]

    def pca_encode(b):
        return (b - mean_us) @ V_kept / S_kept

    def pca_decode(z):
        return mean_us + (z * S_kept) @ V_kept.T

    def sample_extra(n_samp=1):
        eps = np.random.randn(n_samp, S_res.shape[0])
        return (eps * S_res) @ V_res.T

    def extra_cov():
        return V_res @ np.diag(S_res ** 2) @ V_res.T

    return pca_encode, pca_decode, k, sample_extra, extra_cov


# ---------------------------------------------------------------------------
# h-MALA chain loader
# ---------------------------------------------------------------------------
def load_hmala_chains(folder, no_samples, thin, is_delta=False, n_chains=3):
    chains = []
    for i in range(n_chains):
        if is_delta:
            c = np.load(os.path.join(folder, f"chain_{i:03d}", "hmala_samples_delta.npy"))
        else:
            c = np.load(os.path.join(folder, f"chain_{i:03d}", "hmala_samples.npy"))
        chains.append(c.reshape(no_samples, 21, 21, 3, order="F")[::thin, :, :, 0])
    return np.concatenate(chains).reshape(-1, (nx + 1) * (ny + 1))


# ---------------------------------------------------------------------------
# Score conversion: velocity field b_t -> score η_z via trig-interpolant Wronskian
# ---------------------------------------------------------------------------
alpha_t = lambda t: jnp.cos(jnp.pi * t * 0.5)
beta_t  = lambda t: jnp.sin(jnp.pi * t * 0.5)
alpha_dot   = vmap(grad(alpha_t))
beta_dot    = vmap(grad(beta_t))
alpha_t_vmap = vmap(alpha_t)
beta_t_vmap  = vmap(beta_t)


def drift_to_score(drift):
    @eqx.filter_jit
    def score(t, x):
        # t: (n, 1), x: (n, d)
        # Wronskian W = alpha*beta_dot - alpha_dot*beta = pi/2 for trig (constant)
        t_sq = t.squeeze(-1)                        # (n,)
        a_t = alpha_t_vmap(t_sq)[:, None]           # (n, 1)
        b_t = beta_t_vmap(t_sq)[:, None]            # (n, 1)
        ad_t = alpha_dot(t_sq)[:, None]             # (n, 1)
        bd_t = beta_dot(t_sq)[:, None]              # (n, 1)
        W = a_t * bd_t - ad_t * b_t                # (n, 1), = pi/2 for trig
        num = b_t * drift(jnp.hstack([t, x])) - bd_t * x  # (n, d)
        return num / (a_t * W)                     # (n, d)
    return score


# ---------------------------------------------------------------------------
# Simple standard-Gaussian reference sampler (fallback; u0_cond is preferred)
# ---------------------------------------------------------------------------
def standard_gaussian_sampler(key, shape):
    return jax.random.normal(key, shape)


# ---------------------------------------------------------------------------
# Arguments and output directory
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=0, help="Run index for output folder")
args = parser.parse_args()
RANK = args.run_id

output_root = "convergence_results_sde"
output_dir  = os.path.join(output_root, f"run_{RANK:02d}")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
nx = ny     = 20
flat_length = (nx + 1) * (ny + 1)  # 441
n_obs       = 100
nsamples    = 50_000
n_projections = 2048
swd_seed    = 42
EPOCHS      = 650

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
best_hyperparams = {
    "hidden_layer":      256,
    "num_hidden_layers": 3,
    "peak_value":        0.002,
    "weight_decay":      0.0002,
}

HMALA_NO_SAMPLES = 300_000
HMALA_THIN       = 20

COND_LABELS = ["obs", "med", "98"]

# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------
run = wandb.init(
    project="pPoisson - SI - SDE - convergence",
    config={
        "dataset":     "p_poisson_hmala",
        "nsamples":    nsamples,
        "interpolant": "trig_interpolant",
        "sampler":     "SDE",
        "hyperparams": best_hyperparams,
    },
    name=f"run={RANK}_sde_convergence",
)

# ---------------------------------------------------------------------------
# Load h-MALA chains
# ---------------------------------------------------------------------------
print("Loading h-MALA chains...")
hmala_samps = load_hmala_chains("mcmc_main",   HMALA_NO_SAMPLES, HMALA_THIN, is_delta=True)
hmala_med   = load_hmala_chains("mcmc_median", HMALA_NO_SAMPLES, HMALA_THIN)
hmala_98    = load_hmala_chains("mcmc_98",     HMALA_NO_SAMPLES, HMALA_THIN)
print(f"h-MALA (obs): {hmala_samps.shape},  (med): {hmala_med.shape},  (98): {hmala_98.shape}")

hmala_list = [hmala_samps, hmala_med, hmala_98]

# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------
print("Loading training data...")
train_dim = 100_000
ys_all = np.load("training_dataset/solutions_delta.npy")                    # (250000, 100)
us_all = (np.load("training_dataset/parameters_delta.npy")
          .reshape(-1, 21, 21, 3, order="F")[:, :, :, 0]
          .reshape(-1, flat_length))                                          # (250000, 441)
print(f"Full dataset: ys={ys_all.shape}, us={us_all.shape}")

ys = ys_all[:train_dim]
us = us_all[:train_dim]

us_ref = us_all[train_dim : train_dim * 2].copy()
np.random.shuffle(us_ref)

us_test = us_all[train_dim * 2 :].copy()

# ---------------------------------------------------------------------------
# Observation data and normalizer
# ---------------------------------------------------------------------------
yobs     = np.load("data_obs.npy")
yobs_med = np.load("data_50.npy")
yobs_98  = np.load("data_98.npy")

ys_normalizer   = UnitGaussianNormalizer(ys)
ys_normalized   = ys_normalizer.encode()
yobs_normalized = ys_normalizer.encode(yobs)
ymed_normalized = ys_normalizer.encode(yobs_med)
y98_normalized  = ys_normalizer.encode(yobs_98)
ncond_vals = [yobs_normalized, ymed_normalized, y98_normalized]

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
print("Computing PCA on reference parameters...")
pca_encode, pca_decode, k, sample_extra, extra_cov = get_pca_fns(us_ref)
print(f"PCA retains {k} components (99% variance explained)")

us_pca     = jnp.asarray(pca_encode(us))
us_ref_pca = jnp.asarray(pca_encode(us_ref))

u0_cond = jnp.asarray(pca_encode(us_test[:nsamples]))

x1_data = jnp.hstack([jnp.asarray(ys_normalized), us_pca])
x0_data = jnp.hstack([jnp.asarray(ys_normalized), us_ref_pca])

yu_dimension     = (ys_normalized.shape[1], k)   # (100, ~200)
dim              = yu_dimension[0] + yu_dimension[1]
interpolant_args = {"t": None, "x1": None, "x0": None}

# ---------------------------------------------------------------------------
# Base SWDs: prior vs. each h-MALA posterior
# ---------------------------------------------------------------------------
print("Computing base SWDs (prior vs. each h-MALA set)...")
rng_base = np.random.default_rng(swd_seed)
ref_idxs = rng_base.choice(len(us_ref), size=nsamples, replace=False)
us_ref_subset = us_ref[ref_idxs]

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

np.save(os.path.join(output_dir, "base_swd_list.npy"), np.array(base_swd_list))

# ---------------------------------------------------------------------------
# Sample-size list
# ---------------------------------------------------------------------------
sample_no_list = [2 ** i for i in range(1, 15)]
sample_no_list += [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
sample_no_list  = sorted(set(sample_no_list))
n_sizes         = len(sample_no_list)

n_cond        = len(COND_LABELS)
swd_array     = np.zeros((n_sizes, n_cond))
rel_swd_array = np.zeros((n_sizes, n_cond))
u_mean_arrays = [np.zeros((n_sizes, nx + 1, ny + 1)) for _ in COND_LABELS]
u_var_arrays  = [np.zeros((n_sizes, nx + 1, ny + 1)) for _ in COND_LABELS]

sde_solver_args = {
    "solver":   diffrax.Heun(),
    "saveat":   "t1",
    "max_steps": 50000,
}

# ---------------------------------------------------------------------------
# Convergence loop
# ---------------------------------------------------------------------------
t_start = time.time()

for i, sample_no in tqdm(enumerate(sample_no_list), total=n_sizes, desc="sample sizes"):
    print(f"\n{'='*60}")
    print(f"  Training with {sample_no:,} samples  ({i+1}/{n_sizes})")
    print(f"{'='*60}")

    key = random.PRNGKey(i + 1 + RANK)
    key, model_key, sample_key = random.split(key, 3)

    batch_size      = max(1, min(2048, sample_no) - 1)
    steps_per_epoch = int(np.ceil(sample_no / batch_size))
    steps           = steps_per_epoch * EPOCHS

    model = MLP(
        key=model_key,
        dim=dim,
        w=[best_hyperparams["hidden_layer"]] * best_hyperparams["num_hidden_layers"],
        num_layers=best_hyperparams["num_hidden_layers"] + 1,
        activation_fn=jax.nn.gelu,
        time_varying=True,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=best_hyperparams["peak_value"],
        warmup_steps=max(50, steps // 20),
        decay_steps=steps,
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=best_hyperparams["weight_decay"]),
    )

    trainer = NNTrainer(
        target_density=None,
        model=model,
        optimizer=optimizer,
        interpolant=trig_interpolant,
        interpolant_der=trig_interpolant_der,
        reference_sampler=gaussian_reference_sampler,
        loss=vec_field_loss,
        interpolant_args=interpolant_args,
        yu_dimension=yu_dimension,
    )

    trainer.train(
        train_data=x1_data[:sample_no],
        train_dim=sample_no,
        batch_size=batch_size,
        steps=steps,
        x0_data=x0_data[:sample_no],
        print_every=10_000,
    )

    velocity = jax.vmap(trainer.model)
    score    = drift_to_score(velocity)

    print("  Generating samples via SDE (all 3 conditions)...")
    cond_samples_list = conditional_sample(
        velocity=velocity,
        score=score,
        cond_values=ncond_vals,
        yu_dimension=yu_dimension,
        reference_sampler=standard_gaussian_sampler,
        reference_sampler_args=None,
        nsamples=nsamples,
        u0_cond=u0_cond,
        solver_args=sde_solver_args,
        key=sample_key,
    )

    log_dict = {}
    for j, (label, cond_samples, h_flat, base_swd) in enumerate(zip(
        COND_LABELS, cond_samples_list, hmala_list, base_swd_list
    )):
        u_pca_gen  = np.array(cond_samples[:, yu_dimension[0]:])   # (nsamples, k)
        u_flat_gen = np.array(pca_decode(u_pca_gen))               # (nsamples, flat_length)
        extra_noise = sample_extra(nsamples)
        u_flat = u_flat_gen + extra_noise

        u_2d = u_flat.reshape(nsamples, nx + 1, ny + 1)
        u_mean_arrays[j][i] = np.mean(u_2d, axis=0)
        u_var_arrays[j][i]  = np.var(u_2d, axis=0)

        u_flat_jax = jnp.asarray(u_flat)
        h_flat_jax = jnp.asarray(h_flat)

        print(f"  Computing SWD ({label})...")
        swd_val = float(sliced_wasserstein_jax(
            u_flat_jax, h_flat_jax, n_projections=n_projections, seed=swd_seed,
        ))
        swd_array[i, j]     = swd_val
        rel_swd_array[i, j] = swd_val / base_swd

        log_dict[f"relative_swd_{label}"] = rel_swd_array[i, j]
        log_dict[f"swd_{label}"]          = swd_array[i, j]

        print(f"  [{label}] SWD (rel): {rel_swd_array[i, j]:.6f}")

    wandb.log(log_dict, step=sample_no)

    del model, trainer, optimizer, cond_samples_list
    jax.clear_caches()
    gc.collect()

elapsed = time.time() - t_start
print(f"\nTotal elapsed time: {elapsed/60:.1f} min")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
np.save(os.path.join(output_dir, "sample_no_list.npy"), np.array(sample_no_list))
np.save(os.path.join(output_dir, "swd_array.npy"),      swd_array)
np.save(os.path.join(output_dir, "rel_swd_array.npy"),  rel_swd_array)
for j, label in enumerate(COND_LABELS):
    np.save(os.path.join(output_dir, f"u_mean_{label}.npy"), u_mean_arrays[j])
    np.save(os.path.join(output_dir, f"u_var_{label}.npy"),  u_var_arrays[j])
np.save(os.path.join(output_dir, "elapsed_time.npy"), np.array([elapsed]))
print("Results saved. Done.")
