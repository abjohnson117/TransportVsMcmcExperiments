"""
Sample convergence study for the Poisson problem — SDE model.

Trains a stochastic-interpolant SDE (linear) on subsets of size n from the
full training dataset (solutions_grid_delta.npy / parameters_delta.npy),
then evaluates quality of generated posterior samples against h-MALA reference
chains via SWD as n grows.  Three conditioning values are tested: main
observation, median, and 98th-percentile.

Training data layout
--------------------
  solutions_grid_delta.npy : (N, 100)   — noisy 10×10 pointwise observations (y)
  parameters_delta.npy     : (N, 1089)  — 33×33 parameter field (flat)

Hyperparameters: hard-coded (velocity: 512×4, score: 512×4)
Interpolant:     linear with SDE noise schedule gamma_fn(t)
PCA:             whitened PCA (99% variance) with per-sample residual noise injection
"""

import gc
import os
import time
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import diffrax
import wandb
import h5py
from jax import grad, vmap, random
from tqdm.auto import tqdm
from typing import Callable, List
import equinox as eqx

from triangular_transport.flows.sde_flow_trainer import NNSDE
from triangular_transport.flows.loss_functions import vec_field_loss, denoiser_loss
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer
from triangular_transport.flows.dataloaders import gaussian_reference_sampler


# ---------------------------------------------------------------------------
# JAX sliced Wasserstein distance (memory-efficient, supports unequal sizes)
# ---------------------------------------------------------------------------
def sliced_wasserstein_jax(x, y, n_projections=512, seed=42, chunk_size=64, p=2):
    """
    Memory-efficient SWD in JAX.  Supports unequal sample sizes via quantile
    interpolation.  chunk_size=64 uses ~100 MB/kernel.
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
# MLP with residual skip connections
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
# SDE noise schedule and interpolant
# ---------------------------------------------------------------------------
def gamma_fn(t):
    return 5e-3 * jnp.sqrt(2 * (t - t**2) + 1e-8)

gamma_vmap = vmap(gamma_fn)
gammadot   = vmap(grad(gamma_fn))


@vmap
def linear_interpolant(t, x1, x0, z):
    return t * x1 + (1 - t) * x0 + gamma_vmap(t) * z


@vmap
def linear_interpolant_der(t, x1, x0, z):
    return x1 - x0 + gammadot(t) * z


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
        return (eps * S_res) @ V_res.T  # (n_samp, flat_length)

    return pca_encode, pca_decode, k, sample_extra, S_kept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_data_h5(path="data.h5"):
    with h5py.File(path, "r") as f:
        return f["/data"][...]


def load_hmala_chains(folder, n_chains=10, thin=50):
    """Load and concatenate h-MALA chains; returns shape (n_total, flat_length)."""
    chains = []
    for i in range(n_chains):
        arr = np.load(os.path.join(folder, f"chain_{i:03d}", "hmala_samples.npz"))["arr"]
        chains.append(arr[::thin])
    return np.concatenate(chains, axis=0)


# ---------------------------------------------------------------------------
# Arguments and output directory
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=0, help="Run index for output folder")
args = parser.parse_args()
RANK = args.run_id

output_root = "convergence_results_sde2"
output_dir  = os.path.join(output_root, f"run_{RANK:02d}")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
nx = ny      = 33
flat_length  = nx * ny      # 1089
n_obs        = 100          # 10×10 observation grid
train_dim    = 50_000
nsamples     = 20_000
n_projections = 2048
swd_seed     = 42
EPOCHS       = 700        # p-poisson ODE used 1300; banana ODE→SDE ratio is 1.25×, so 1300*1.25≈1625
# MIN_STEPS    = 10_000       # floor: ensures adequate steps for small n where steps_per_epoch=2
HMALA_THIN   = 50           # thin factor per chain; 100000/50 × 10 chains = 20000

COND_LABELS  = ["obs", "med", "98"]

# ---------------------------------------------------------------------------
# Hyperparameters (fixed, well-tuned for Poisson SDE)
# ---------------------------------------------------------------------------
best_hyperparams = {
    "v_hidden":  256,
    "v_layers":  3,
    "v_peak_lr": 2e-3,
    "s_hidden":  512,
    "s_layers":  3,
    "s_peak_lr": 3e-4,
}

# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------
run = wandb.init(
    project="Poisson - SDE v hMALA - convergence",
    config={
        "dataset":     "poisson_hmala",
        "train_dim":   train_dim,
        "nsamples":    nsamples,
        "interpolant": "linear_sde",
        "hyperparams": best_hyperparams,
    },
    name=f"run={RANK}_sde_convergence",
)

# ---------------------------------------------------------------------------
# Load h-MALA chains (all three conditioning values)
# ---------------------------------------------------------------------------
print("Loading h-MALA chains...")
hmala_obs = load_hmala_chains("mcmc_main_converge_samps", thin=HMALA_THIN)
hmala_med = load_hmala_chains("mcmc_med_converge_samps",  thin=HMALA_THIN)
hmala_98  = load_hmala_chains("mcmc_98_converge_samps",   thin=HMALA_THIN)
print(f"h-MALA shapes — obs: {hmala_obs.shape}, med: {hmala_med.shape}, 98: {hmala_98.shape}")
hmala_list = [hmala_obs, hmala_med, hmala_98]

# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------
print("Loading training data...")
ys_all = np.load("training_dataset/solutions_grid_delta.npy")
us_all = np.load("training_dataset/parameters_delta.npy").reshape(-1, flat_length)
print(f"Full dataset: ys={ys_all.shape}, us={us_all.shape}")

ys     = ys_all[:train_dim]
us     = us_all[:train_dim]
us_ref = us_all[train_dim : train_dim * 2].copy()
np.random.shuffle(us_ref)
us_test = us_all[train_dim * 2 : train_dim * 2 + nsamples].copy()

# ---------------------------------------------------------------------------
# Observation data and normalizer
# ---------------------------------------------------------------------------
# yobs     = read_data_h5()
yobs = np.load("data_obs.npy")
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
pca_encode, pca_decode, k, sample_extra, S_kept = get_pca_fns(us_ref)
print(f"PCA retains {k} components (99% variance explained)")

us_pca     = jnp.asarray(pca_encode(us))
us_ref_pca = jnp.asarray(pca_encode(us_ref))
u0_cond    = jnp.asarray(pca_encode(us_test))  # initial conditions for ODE

# D_mask: zeros on observation dims, sqrt(S) on PCA dims (whitened noise scale)
D_mask = jnp.concatenate([jnp.zeros(n_obs), jnp.sqrt(S_kept)])

# Joint (y, u_pca) datasets for the SDE trainer
x1_data = jnp.hstack([jnp.asarray(ys_normalized), us_pca])
x0_data = jnp.hstack([jnp.asarray(ys_normalized), us_ref_pca])

yu_dimension          = (n_obs, k)
dim                   = n_obs + k
interpolant_args      = {"t": None, "x1": None, "x0": None, "z": None}
reference_sampler_args = {"D_mask": D_mask}
# solver_args           = {"saveat": "t1", "D_mask": D_mask}
solver_args = {"D_mask": D_mask, "saveat": "t1", "eps": 5e-3, "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6)}

# ---------------------------------------------------------------------------
# Base SWD: prior (reference parameters) vs. each h-MALA posterior
# ---------------------------------------------------------------------------
print("Computing base SWDs (prior vs. each h-MALA set, original space)...")
rng_base   = np.random.default_rng(swd_seed)
ref_idxs   = rng_base.choice(len(us_ref), size=nsamples, replace=False)
us_ref_sub = us_ref[ref_idxs]

base_swd_list = []
for label, h_flat in zip(COND_LABELS, hmala_list):
    bswd = float(sliced_wasserstein_jax(
        jnp.asarray(us_ref_sub),
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
sample_no_list = [2 ** i for i in range(1, 15)]          # 2 … 16384
sample_no_list += [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
sample_no_list  = sorted(set(sample_no_list))
n_sizes         = len(sample_no_list)

n_cond        = len(COND_LABELS)
swd_array     = np.zeros((n_sizes, n_cond))
rel_swd_array = np.zeros((n_sizes, n_cond))
u_mean_arrays = [np.zeros((n_sizes, nx, ny)) for _ in COND_LABELS]
u_var_arrays  = [np.zeros((n_sizes, nx, ny)) for _ in COND_LABELS]

# ---------------------------------------------------------------------------
# Convergence loop
# ---------------------------------------------------------------------------
t_start = time.time()

for i, sample_no in tqdm(enumerate(sample_no_list), total=n_sizes, desc="sample sizes"):
    print(f"\n{'='*60}")
    print(f"  Training with {sample_no:,} samples  ({i+1}/{n_sizes})")
    print(f"{'='*60}")

    key = random.PRNGKey(i + 1 + RANK)
    key, vkey, skey = random.split(key, 3)

    batch_size      = max(1, min(2048, sample_no) - 1)
    steps_per_epoch = int(np.ceil(sample_no / batch_size))
    # steps           = max(steps_per_epoch * EPOCHS, MIN_STEPS)
    steps = steps_per_epoch * EPOCHS

    velocity = MLP(
        key=vkey,
        dim=dim,
        w=[best_hyperparams["v_hidden"]] * best_hyperparams["v_layers"],
        num_layers=best_hyperparams["v_layers"] + 1,
        activation_fn=jax.nn.gelu,
        time_varying=True,
    )
    score = MLP(
        key=skey,
        dim=dim,
        w=[best_hyperparams["s_hidden"]] * best_hyperparams["s_layers"],
        num_layers=best_hyperparams["s_layers"] + 1,
        activation_fn=jax.nn.gelu,
        time_varying=True,
    )

    v_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=best_hyperparams["v_peak_lr"],
        warmup_steps=max(50, steps // 20),
        decay_steps=steps,
        end_value=1e-5,
    )
    s_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=best_hyperparams["s_peak_lr"],
        warmup_steps=max(50, steps // 20),
        decay_steps=steps,
        end_value=1e-5,
    )
    v_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(v_schedule))
    s_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(s_schedule))

    trainer = NNSDE(
        target_density=None,
        velocity=velocity,
        score=score,
        v_optimizer=v_optimizer,
        s_optimizer=s_optimizer,
        interpolant=linear_interpolant,
        interpolant_der=linear_interpolant_der,
        reference_sampler=gaussian_reference_sampler,
        v_loss=vec_field_loss,
        s_loss=denoiser_loss,
        interpolant_args=interpolant_args,
        yu_dimension=yu_dimension,
        reference_sampler_args=reference_sampler_args,
    )

    trainer.train(
        x1_data=x1_data[:sample_no],
        train_dim=sample_no,
        batch_size=batch_size,
        steps=steps,
        x0_data=x0_data[:sample_no],
        print_every=10_000,
    )

    # -- Generate posterior samples for all three conditioning values ---------
    print("  Generating samples (all 3 conditions)...")
    cond_samples_list = trainer.conditional_sample(
        cond_values=ncond_vals,
        u0_cond=u0_cond,
        nsamples=nsamples,
        gamma=gamma_fn,
        solver_args=solver_args,
    )

    # -- Per-condition metrics ------------------------------------------------
    log_dict = {}
    for j, (label, cond_samples, h_flat, base_swd) in enumerate(zip(
        COND_LABELS, cond_samples_list, hmala_list, base_swd_list
    )):
        # Extract u-part from joint (y, u) output
        u_pca_gen = np.array(cond_samples[:, n_obs:])   # (nsamples, k)

        # PCA decode + per-sample residual variance injection
        u_flat_gen  = np.array(pca_decode(u_pca_gen))   # (nsamples, flat_length)
        extra_noise = sample_extra(nsamples)
        u_flat      = u_flat_gen + extra_noise           # (nsamples, flat_length)

        # Posterior mean and variance on the 33×33 grid
        u_2d = u_flat.reshape(nsamples, nx, ny)
        u_mean_arrays[j][i] = np.mean(u_2d, axis=0)
        u_var_arrays[j][i]  = np.var(u_2d, axis=0)

        # SWD in original (non-PCA) space
        print(f"  Computing SWD ({label})...")
        swd_val = float(sliced_wasserstein_jax(
            jnp.asarray(u_flat),
            jnp.asarray(h_flat),
            n_projections=n_projections,
            seed=swd_seed,
        ))
        swd_array[i, j]     = swd_val
        rel_swd_array[i, j] = swd_val / base_swd

        log_dict[f"relative_swd_{label}"] = rel_swd_array[i, j]
        log_dict[f"swd_{label}"]          = swd_array[i, j]

        print(f"  [{label}] SWD={swd_val:.6f}  rel={rel_swd_array[i, j]:.6f}")

    wandb.log(log_dict, step=sample_no)

    del trainer, velocity, score, v_optimizer, s_optimizer, cond_samples_list
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
